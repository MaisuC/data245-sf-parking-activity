"""Part 1 inside Part 2: the next-hour forecast as a clue for the arrival-time model.

For every block and hour in the arrival-time area, this module builds the Part 1
dataset (new paid sessions per meter in the hour, labelled Low / Medium / High by
training-only terciles) with the same features as src/train.py, fits the Part 1
model type (KNN, k chosen on the calibration dates) and returns, for each block and
hour, the forecast probabilities of Low, Medium and High activity.

Only the past is used:
  * Part 1 features for hour H use completed hours before H (lags 1, 2, 24, 168 and a
    3-hour mean), so the forecast for H is available at H:00.
  * Part 2 decisions made during hour H use the forecast for hour H.
  * Training rows get out-of-fold forecasts (folds by date), so a training example
    never sees a forecast fitted on its own label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import FEATURES, NUMERIC, assign_labels, make_pipeline  # noqa: E402

KNN_K = (7, 21, 51)            # the configurations Part 1 compares
FORECAST_COLUMNS = ['p1_low', 'p1_medium', 'p1_high']


def hourly_panel(sessions, meters, blocks, days):
    """Part 1 panel for the arrival-time area: one row per block and hour."""
    ns = sessions[sessions.meter_event_type.eq('NS')]
    counts = ns.groupby(['blockface_id', ns.start.dt.floor('h')]).size().rename('new_session_count')
    all_events = sessions.groupby(['blockface_id', sessions.start.dt.floor('h')]).size().rename('all_events')
    hours = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq='h', inclusive='left')
    info = blocks.set_index('blockface_id')[['supply', 'latitude', 'longitude']]
    grid = pd.MultiIndex.from_product([info.index, hours], names=['blockface_id', 'target_start'])
    panel = pd.concat([counts, all_events], axis=1).rename_axis(['blockface_id', 'target_start'])
    panel = panel.reindex(grid, fill_value=0).reset_index().join(info, on='blockface_id')
    panel['date'] = panel.target_start.dt.normalize()
    covered = panel.groupby(['blockface_id', 'date']).all_events.transform('sum').gt(0)
    panel.loc[~covered, ['new_session_count', 'all_events']] = np.nan
    panel['target_pressure'] = panel.new_session_count / panel.supply
    grouped = panel.groupby('blockface_id', sort=False).target_pressure
    for lag in [1, 2, 24, 168]:
        panel[f'pressure_lag{lag}'] = grouped.shift(lag)
    panel['pressure_rolling3'] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    hour, dow = panel.target_start.dt.hour, panel.target_start.dt.dayofweek
    panel['hour_sin'], panel['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    panel['dow_sin'], panel['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    panel['is_saturday'] = dow.eq(5).astype(int)
    panel['hour'], panel['day_of_week'] = hour, dow
    return panel


def fit_forecast(panel, split_dates):
    """Fit Part 1's model on training dates; return forecasts for every usable block-hour.

    split_dates: dict with 'train' and 'calibration' lists of dates (Part 2's split).
    """
    window = panel.hour.between(8, 18) & panel.day_of_week.le(5)
    data = panel.loc[window].dropna(subset=NUMERIC + ['target_pressure']).copy()
    train_days = set(pd.to_datetime(list(split_dates['train'])))
    cal_days = set(pd.to_datetime(list(split_dates['calibration'])))
    train = data[data.date.isin(train_days)]
    thresholds = train.target_pressure.quantile([1 / 3, 2 / 3]).to_numpy()
    data['target_class'] = assign_labels(data.target_pressure, thresholds)
    train, cal = data[data.date.isin(train_days)], data[data.date.isin(cal_days)]

    # Choose k on the calibration dates, like Part 1 chooses on validation.
    scores = {}
    for k in KNN_K:
        model = make_pipeline(KNeighborsClassifier(n_neighbors=k, weights='distance'))
        model.fit(train[FEATURES], train.target_class)
        scores[k] = f1_score(cal.target_class, model.predict(cal[FEATURES]), average='macro')
    k = max(scores, key=scores.get)
    final = make_pipeline(KNeighborsClassifier(n_neighbors=k, weights='distance'))
    final.fit(train[FEATURES], train.target_class)

    proba = pd.DataFrame(final.predict_proba(data[FEATURES]), index=data.index, columns=FORECAST_COLUMNS)
    # Out-of-fold forecasts for training rows (folds by date).
    folds = GroupKFold(n_splits=5)
    for fit_idx, pred_idx in folds.split(train, groups=train.date):
        part = make_pipeline(KNeighborsClassifier(n_neighbors=k, weights='distance'))
        part.fit(train.iloc[fit_idx][FEATURES], train.iloc[fit_idx].target_class)
        rows = train.index[pred_idx]
        proba.loc[rows] = part.predict_proba(train.loc[rows, FEATURES])
    out = data[['blockface_id', 'target_start', 'date', 'target_class']].join(proba)

    test = data[~data.date.isin(train_days | cal_days)]
    seasonal = train.groupby(['blockface_id', 'day_of_week', 'hour']).target_class.agg(
        lambda s: s.value_counts().idxmax())
    key = pd.MultiIndex.from_frame(test[['blockface_id', 'day_of_week', 'hour']])
    seasonal_pred = seasonal.reindex(key).fillna(train.target_class.mode()[0]).to_numpy()
    quality = {'k_selected_on_calibration': int(k),
               'calibration_macro_f1_by_k': {str(a): round(float(b), 4) for a, b in scores.items()},
               'thresholds_training_only': [round(float(x), 4) for x in thresholds],
               'test_macro_f1_knn': round(float(f1_score(test.target_class, final.predict(test[FEATURES]),
                                                         average='macro')), 4),
               'test_macro_f1_seasonal_mode': round(float(f1_score(test.target_class, seasonal_pred,
                                                                   average='macro')), 4),
               'block_hours': {'train': len(train), 'calibration': len(cal), 'test': len(test)}}
    return out, quality


def add_forecast(examples, forecasts, priors):
    """Attach the forecast for the hour that contains each decision time."""
    hour = examples.decision_time.dt.floor('h')
    f = forecasts.set_index(['blockface_id', 'target_start'])[FORECAST_COLUMNS]
    key = pd.MultiIndex.from_arrays([examples.blockface_id, hour])
    values = f.reindex(key).to_numpy()
    examples = examples.copy()
    for i, col in enumerate(FORECAST_COLUMNS):
        examples[col] = values[:, i]
        examples[col] = examples[col].fillna(priors[i])
    return examples
