"""Downtown arrival-time model: will a space with no active payment exist when I arrive?

Integrates AlaaTarekA/sf-parking-availability (target, calibration, replay, choice
sets) with this repo's SF payment pipeline, calendar features, seasonal baseline
and date-based split.

Run after src/downtown_supply.py and src/download_downtown.py:
    python src/train_downtown.py
Outputs: results/downtown/*.csv, *.json, results/figures/downtown_*.png

Pre-declared primary model: Random Forest with feature set C, isotonic calibration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.downtown_pipeline import (FEATURE_SETS, HORIZONS, add_seasonal, assign_splits,  # noqa: E402
                                   block_panel, build_examples, load_sessions)
from src.downtown_supply import distances  # noqa: E402

RAW, PROCESSED = ROOT / 'data' / 'raw', ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results' / 'downtown'
FIGURES = ROOT / 'results' / 'figures'
SEED, BUSY, BOOT = 245, 0.85, 2000
PRIMARY = ('Random Forest', 'C')
REPLAY_H, RADIUS_M, DRIVERS, WALK_M_PER_MIN, FAIL_COST_MIN = 10, 400, 10_000, 80.0, (3, 6, 10)


def models():
    return {'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_samples=0.5,
                                                    n_jobs=-1, random_state=SEED),
            'Gradient Boosting': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                random_state=SEED)}


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def scores(y, p, busy):
    out = {'brier': brier_score_loss(y, p), 'ece': ece(y, p),
           'brier_busy': brier_score_loss(y[busy], p[busy]) if busy.any() else np.nan}
    if len(np.unique(y)) > 1 and len(np.unique(p)) > 1:
        out['auc'] = roc_auc_score(y, p)
    return {k: round(float(v), 4) for k, v in out.items()}


def day_bootstrap(test, p_a, p_b, rng):
    """Brier(a) - Brier(b) with a 95% interval from resampling test days."""
    sq_a, sq_b = (p_a - test.y.to_numpy()) ** 2, (p_b - test.y.to_numpy()) ** 2
    frame = pd.DataFrame({'d': test.date.to_numpy(), 'diff': sq_a - sq_b})
    per_day = frame.groupby('d')['diff'].agg(['sum', 'size'])
    draws = []
    for _ in range(BOOT):
        s = per_day.sample(len(per_day), replace=True, random_state=int(rng.integers(1e9)))
        draws.append(s['sum'].sum() / s['size'].sum())
    return {'difference': round(float(frame['diff'].mean()), 5),
            'ci95': [round(float(np.quantile(draws, .025)), 5), round(float(np.quantile(draws, .975)), 5)],
            'test_days': int(len(per_day))}


def evaluate(data, h):
    train, cal, test = (data[data.split.eq(s)] for s in ('train', 'calibration', 'test'))
    y, busy = test.y.to_numpy(), (test.occ_now >= BUSY).to_numpy()
    rows = [{'horizon_min': h, 'model': 'Persistence (unpaid now stays unpaid)', 'features': '-',
             **scores(y, test.any_unpaid_now.to_numpy().astype(float), busy)},
            {'horizon_min': h, 'model': 'Seasonal rate (block x weekday x hour)', 'features': '-',
             **scores(y, test.seasonal_rate.to_numpy(), busy)}]
    probs = {}
    for fs, cols in FEATURE_SETS.items():
        for name, model in models().items():
            model.fit(train[cols], train.y)
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic').fit(cal[cols], cal.y)
            p = calibrated.predict_proba(test[cols])[:, 1]
            s = scores(y, p, busy)
            s['ece_before_calibration'] = round(ece(y, model.predict_proba(test[cols])[:, 1]), 4)
            rows.append({'horizon_min': h, 'model': name, 'features': fs, **s})
            probs[(name, fs)] = p
    return pd.DataFrame(rows), probs, test


def replay(test, p, blocks, rng):
    dist = pd.DataFrame(distances(blocks.latitude.to_numpy(), blocks.longitude.to_numpy()),
                        index=blocks.blockface_id, columns=blocks.blockface_id)
    test = test.assign(p=p)
    snaps = {t: g.set_index('blockface_id') for t, g in test.groupby('decision_time')}
    busy = test[test.occ_now >= BUSY]
    if busy.empty:
        return pd.DataFrame()
    picks = busy.sample(DRIVERS, replace=len(busy) < DRIVERS, random_state=SEED)
    out = []
    for _, d in picks.iterrows():
        snap = snaps[d.decision_time]
        near = dist.loc[d.blockface_id]
        cand = near[(near <= RADIUS_M) & near.index.isin(snap.index)]
        snap = snap.loc[cand.index]
        walk, ok = cand.to_numpy(), snap.unpaid_future.to_numpy() >= 1
        free_now = snap.any_unpaid_now.to_numpy() == 1
        dest = int(np.argmin(walk))
        choice = {'Destination block': dest,
                  'Nearest block with an unpaid space now': int(np.argmin(np.where(free_now, walk, np.inf)))
                  if free_now.any() else dest}
        for c in FAIL_COST_MIN:
            choice[f'Model recommendation (fail cost {c} min)'] = int(np.argmin(
                walk + (1 - snap.p.to_numpy()) * c * WALK_M_PER_MIN))
        out += [{'strategy': k, 'success': bool(ok[v]), 'walk_m': float(walk[v])} for k, v in choice.items()]
    res = pd.DataFrame(out).groupby('strategy').agg(first_try_success=('success', 'mean'),
                                                     avg_walk_m=('walk_m', 'mean'))
    return res.round({'first_try_success': 3, 'avg_walk_m': 1}).sort_values('first_try_success').reset_index()


def figures(metrics, rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    keep = metrics[metrics.features.isin(['C', '-'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], 'brier', 'All blocks'), (axes[1], 'brier_busy', 'Busy blocks (85%+ paid)')):
        for name, g in keep.groupby('model'):
            ax.plot(g.horizon_min, g[col], marker='o', label=name)
        ax.set(title=f'Test Brier score, {title}', xlabel='Minutes until arrival',
               ylabel='Brier score (lower is better)', xticks=list(HORIZONS))
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / 'downtown_brier_by_horizon.png', dpi=160); plt.close(fig)
    if not rep.empty:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.barh(rep.strategy, rep.first_try_success * 100,
                color=['#1F8A80' if s.startswith('Model') else '#9AA5A8' for s in rep.strategy])
        for i, r in enumerate(rep.itertuples()):
            ax.text(r.first_try_success * 100 + .8, i, f'{r.first_try_success:.0%} · walk {r.avg_walk_m:.0f} m',
                    va='center', fontsize=9)
        ax.set(xlim=(0, 110), xlabel='Found an unpaid space on the first block tried (%)',
               title=f'Replay: {DRIVERS:,} drivers heading to a busy Downtown block, arriving in {REPLAY_H} min')
        fig.tight_layout(); fig.savefig(FIGURES / 'downtown_replay.png', dpi=160); plt.close(fig)


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    meters = pd.read_csv(RAW / 'downtown_meters.csv', dtype={'post_id': str, 'blockface_id': str})
    blocks = pd.read_csv(PROCESSED / 'downtown_blocks.csv', dtype={'blockface_id': str})
    manifest = json.loads((RAW / 'downtown_manifest.json').read_text())
    sessions = load_sessions(pd.read_csv(RAW / 'downtown_sessions.csv.gz', dtype={'post_id': str}), meters)
    days = pd.date_range(manifest['start_inclusive'], manifest['end_exclusive'], freq='D', inclusive='left')
    panel = block_panel(sessions, meters, blocks, days)
    rng = np.random.default_rng(SEED)
    all_metrics, summary, rep = [], {'sessions_used': len(sessions)}, pd.DataFrame()
    for h in HORIZONS:
        data = add_seasonal(assign_splits(build_examples(panel, blocks, h)))
        metrics, probs, test = evaluate(data, h)
        all_metrics.append(metrics)
        p_main = probs[PRIMARY]
        summary[f'{h}min'] = {
            'rows': data.split.value_counts().to_dict(),
            'split_dates': {s: [str(g.date.min().date()), str(g.date.max().date())] for s, g in data.groupby('split')},
            'base_rate_test': round(float(test.y.mean()), 3),
            'primary_vs_persistence': day_bootstrap(test, p_main, test.any_unpaid_now.to_numpy().astype(float), rng),
            'primary_vs_seasonal': day_bootstrap(test, p_main, test.seasonal_rate.to_numpy(), rng),
            'rq2_C_vs_B_random_forest': day_bootstrap(test, p_main, probs[('Random Forest', 'B')], rng),
        }
        if h == REPLAY_H:
            rep = replay(test, p_main, blocks, rng)
            rep.to_csv(RESULTS / 'replay_10min.csv', index=False)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(RESULTS / 'test_metrics.csv', index=False)
    summary.update({'primary_model': f'{PRIMARY[0]}, feature set {PRIMARY[1]}, isotonic calibration',
                    'note': 'Negative difference = primary model has lower (better) Brier score.',
                    'target': 'At least one meter with no active payment at arrival; not verified free parking.',
                    'runtime_seconds': round(time.time() - t0, 1)})
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    figures(metrics, rep)
    print(metrics.to_string(index=False))
    print(rep.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
