"""Downtown arrival-time model: will a space with no active payment exist when I arrive?

Integrates AlaaTarekA/sf-parking-availability (target, calibration, replay, choice
sets) with this repo's SF payment pipeline, calendar features, seasonal baseline
and date-based split.

Run after src/downtown_supply.py and src/download_downtown.py:
    python src/train_downtown.py
Outputs: results/downtown/*.csv, *.json, results/figures/downtown_*.png

Pre-declared primary model: Random Forest with feature set C, isotonic calibration.
Integration (Part 1 -> Part 2): feature set D adds Part 1's next-hour forecast
(src/hourly_forecast.py) and is compared with C by the same day bootstrap.

One question, 5 minutes to 1 hour ahead (horizon sweep): at arrival times of
5, 10, 15, 30, 45 and 60 minutes, one Random Forest is trained with planning clues
only (Part 1: weekly rhythm, recent days, next-hour forecast), live clues only
(Part 2: the street now, paid time ending) and both. Removing a group and seeing
how much worse the model gets shows what that group adds at each horizon. All
horizons are scored on the same test decision times (09:00-17:00).
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
from src.downtown_pipeline import (FEATURE_SETS, HORIZONS, SWEEP_HORIZONS, SWEEP_LAST_DECISION,  # noqa: E402
                                   SWEEP_SETS, add_seasonal, assign_splits, block_panel,
                                   build_examples, load_sessions)
from src.downtown_supply import distances  # noqa: E402
from src.hourly_forecast import FORECAST_COLUMNS, add_forecast, fit_forecast, hourly_panel  # noqa: E402

RAW, PROCESSED = ROOT / 'data' / 'raw', ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results' / 'downtown'
FIGURES = ROOT / 'results' / 'figures'
SEED, BUSY, BOOT = 245, 0.85, 2000
PRIMARY = ('Random Forest', 'C')
WITH_PART1 = ('Random Forest', 'D')
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


def sweep(data, h, rng):
    """Planning clues vs live clues vs both, one Random Forest each, at horizon h."""
    train, cal, test = (data[data.split.eq(s)] for s in ('train', 'calibration', 'test'))
    minute = test.decision_time.dt.hour * 60 + test.decision_time.dt.minute
    test = test[minute <= SWEEP_LAST_DECISION]
    y, busy = test.y.to_numpy(), (test.occ_now >= BUSY).to_numpy()
    rows = [{'horizon_min': h, 'clues': 'Persistence (unpaid now stays unpaid)',
             **scores(y, test.any_unpaid_now.to_numpy().astype(float), busy)},
            {'horizon_min': h, 'clues': 'Seasonal rate (block x weekday x hour)',
             **scores(y, test.seasonal_rate.to_numpy(), busy)}]
    probs = {}
    for name, cols in SWEEP_SETS.items():
        model = models()['Random Forest'].fit(train[cols], train.y)
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic').fit(cal[cols], cal.y)
        probs[name] = calibrated.predict_proba(test[cols])[:, 1]
        rows.append({'horizon_min': h, 'clues': name, **scores(y, probs[name], busy)})
    both = probs['Both (one model)']
    added = {'live_clues_add': day_bootstrap(test, both, probs['Planning clues only'], rng),
             'planning_clues_add': day_bootstrap(test, both, probs['Live clues only'], rng),
             'base_rate_test': round(float(y.mean()), 3), 'test_rows': int(len(test))}
    return pd.DataFrame(rows), added


def replay(test, p, blocks, rng, p_part1=None):
    dist = pd.DataFrame(distances(blocks.latitude.to_numpy(), blocks.longitude.to_numpy()),
                        index=blocks.blockface_id, columns=blocks.blockface_id)
    test = test.assign(p=p, p_d=p if p_part1 is None else p_part1)
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
            if p_part1 is not None:
                choice[f'Model + next-hour forecast (fail cost {c} min)'] = int(np.argmin(
                    walk + (1 - snap.p_d.to_numpy()) * c * WALK_M_PER_MIN))
        out += [{'strategy': k, 'success': bool(ok[v]), 'walk_m': float(walk[v])} for k, v in choice.items()]
    res = pd.DataFrame(out).groupby('strategy').agg(first_try_success=('success', 'mean'),
                                                     avg_walk_m=('walk_m', 'mean'))
    return res.round({'first_try_success': 3, 'avg_walk_m': 1}).sort_values('first_try_success').reset_index()


def figures(metrics, rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    keep = metrics[metrics.features.isin(['C', 'D', '-'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], 'brier', 'All blocks'), (axes[1], 'brier_busy', 'Busy blocks (85%+ paid)')):
        for (name, fs), g in keep.groupby(['model', 'features']):
            ax.plot(g.horizon_min, g[col], marker='o', label=name if fs == '-' else f'{name} ({fs})')
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


def sweep_figure(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    wide = table.pivot(index='horizon_min', columns='clues', values='brier')
    both = wide['Both (one model)']
    live_add = (wide['Planning clues only'] - both) / wide['Planning clues only'] * 100
    plan_add = (wide['Live clues only'] - both) / wide['Live clues only'] * 100
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    style = {'Both (one model)': ('#1F8A80', '-'), 'Live clues only': ('#D9822B', '--'),
             'Planning clues only': ('#3D5A80', '--'), 'Persistence (unpaid now stays unpaid)': ('#9AA5A8', ':'),
             'Seasonal rate (block x weekday x hour)': ('#6C757D', ':')}
    for name, (color, ls) in style.items():
        axes[0].plot(wide.index, wide[name], marker='o', color=color, ls=ls, label=name)
    axes[0].set(title='Test Brier score by arrival time', xlabel='Minutes until arrival',
                ylabel='Brier score (lower is better)', xticks=list(SWEEP_HORIZONS))
    axes[0].legend(fontsize=8)
    x = np.arange(len(wide.index))
    axes[1].bar(x - .2, live_add, .4, color='#D9822B', label='Live clues (Part 2)')
    axes[1].bar(x + .2, plan_add, .4, color='#3D5A80', label='Planning clues (Part 1)')
    axes[1].set(xticks=x, xticklabels=wide.index, xlabel='Minutes until arrival',
                ylabel='Error removed by adding the group (%)',
                title='What each kind of clue adds to the other')
    axes[1].axhline(0, color='black', lw=.6)
    axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES / 'downtown_horizon_sweep.png', dpi=160); plt.close(fig)


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
    first = add_seasonal(assign_splits(build_examples(panel, blocks, HORIZONS[0])))
    split_dates = {s: sorted(g.date.unique()) for s, g in first.groupby('split')}
    forecasts, part1_quality = fit_forecast(hourly_panel(sessions, meters, blocks, days), split_dates)
    train_fc = forecasts[forecasts.date.isin(pd.to_datetime(list(split_dates['train'])))]
    priors = [float((train_fc.target_class == c).mean()) for c in range(3)]
    forecasts.to_csv(RESULTS / 'part1_hourly_forecasts.csv.gz', index=False, compression='gzip')
    summary['part1_forecast_in_area'] = part1_quality
    prepared = {}
    for h in HORIZONS:
        data = add_seasonal(assign_splits(build_examples(panel, blocks, h)))
        data = add_forecast(data, forecasts, priors)
        prepared[h] = data
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
            'integration_D_vs_C_random_forest': day_bootstrap(test, probs[WITH_PART1], p_main, rng),
        }
        if h == REPLAY_H:
            rep = replay(test, p_main, blocks, rng, probs[WITH_PART1])
            rep.to_csv(RESULTS / 'replay_10min.csv', index=False)
    sweep_rows, summary['horizon_sweep'] = [], {
        'note': 'difference = Brier(both) - Brier(without the group); negative = the group helps.',
        'test_decisions': f'09:00 to {SWEEP_LAST_DECISION // 60}:00, same for every horizon'}
    for h in SWEEP_HORIZONS:
        data = prepared.pop(h, None)
        if data is None:
            data = add_forecast(add_seasonal(assign_splits(build_examples(panel, blocks, h))), forecasts, priors)
        table, added = sweep(data, h, rng)
        sweep_rows.append(table)
        summary['horizon_sweep'][f'{h}min'] = added
        print(f'sweep {h} min done ({time.time() - t0:.0f}s)', flush=True)
    sweep_table = pd.concat(sweep_rows, ignore_index=True)
    sweep_table.to_csv(RESULTS / 'horizon_sweep.csv', index=False)
    sweep_figure(sweep_table)
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
    print(sweep_table.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
