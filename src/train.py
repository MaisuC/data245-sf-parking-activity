"""One-hour-ahead payment-activity forecasting with chronological evaluation."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.build/matplotlib'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / 'data/raw', ROOT / 'data/processed'
RESULTS, FIGURES = ROOT / 'results', ROOT / 'results/figures'
CLASS_NAMES = ['Low payment activity', 'Medium payment activity', 'High payment activity']
CALENDAR = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_saturday']
STATIC = ['supply', 'latitude', 'longitude']
HISTORY = ['pressure_lag1', 'pressure_lag2', 'pressure_lag24',
           'pressure_lag168', 'pressure_rolling3']
NUMERIC = CALENDAR + STATIC + HISTORY
CATEGORICAL = ['blockface_id']
FEATURES = NUMERIC + CATEGORICAL
SEED = 245


def assign_labels(values, thresholds):
    # Inclusive upper boundaries make zero remain in the low class.
    return np.searchsorted(np.asarray(thresholds), np.asarray(values), side='left')


def build_dataset():
    manifest = json.loads((RAW / 'download_manifest.json').read_text())
    meters = pd.read_csv(RAW / 'selected_meters.csv', dtype={'post_id': str, 'blockface_id': str})
    events = pd.read_csv(RAW / 'hourly_meter_events.csv', dtype={'post_id': str})
    if events.duplicated(['post_id', 'day', 'hour', 'meter_event_type']).any():
        raise ValueError('Duplicate aggregate rows would double-count payments.')
    events['event_count'] = pd.to_numeric(events.event_count)
    events['target_start'] = pd.to_datetime(events.day) + pd.to_timedelta(events.hour.astype(int), unit='h')
    events = events.merge(meters[['post_id', 'blockface_id']], on='post_id', validate='many_to_one')
    events['new_sessions'] = np.where(events.meter_event_type.eq('NS'), events.event_count, 0)
    hourly = events.groupby(['blockface_id', 'target_start']).agg(
        all_events=('event_count', 'sum'), new_session_count=('new_sessions', 'sum'))
    supply = meters.groupby('blockface_id').agg(supply=('post_id', 'nunique'),
        latitude=('latitude', 'mean'), longitude=('longitude', 'mean'),
        neighborhood=('analysis_neighborhood', 'first'), street_name=('street_name', 'first'))
    hours = pd.date_range(manifest['start_inclusive'], manifest['end_exclusive'], freq='h', inclusive='left')
    grid = pd.MultiIndex.from_product([supply.index, hours], names=['blockface_id', 'target_start'])
    panel = hourly.reindex(grid, fill_value=0).reset_index().merge(supply, on='blockface_id', validate='many_to_one')
    panel['date'] = panel.target_start.dt.normalize()
    # An entirely silent block-day may be missing data. Do not label it as empty parking.
    covered = panel.groupby(['blockface_id', 'date']).all_events.transform('sum').gt(0)
    panel['covered_block_day'] = covered
    panel.loc[~covered, ['new_session_count', 'all_events']] = np.nan
    panel['target_pressure'] = panel.new_session_count / panel.supply
    grouped = panel.groupby('blockface_id', sort=False).target_pressure
    for lag in [1, 2, 24, 168]:
        panel[f'pressure_lag{lag}'] = grouped.shift(lag)
    panel['pressure_rolling3'] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    panel['latest_observation_end'] = panel.target_start
    panel['latest_source_hour'] = panel.target_start - pd.Timedelta(hours=1)
    panel['hour'] = panel.target_start.dt.hour
    panel['day_of_week'] = panel.target_start.dt.dayofweek
    panel['hour_sin'] = np.sin(2 * np.pi * panel.hour / 24)
    panel['hour_cos'] = np.cos(2 * np.pi * panel.hour / 24)
    panel['dow_sin'] = np.sin(2 * np.pi * panel.day_of_week / 7)
    panel['dow_cos'] = np.cos(2 * np.pi * panel.day_of_week / 7)
    panel['is_saturday'] = panel.day_of_week.eq(5).astype(int)
    # A fixed analytical window, not an assertion about every meter's charging schedule.
    window = panel.hour.between(9, 18) & panel.day_of_week.le(5)
    dataset = panel.loc[window].dropna(subset=NUMERIC + ['target_pressure']).copy()
    dataset = dataset.sort_values(['target_start', 'blockface_id']).reset_index(drop=True)
    if len(dataset) < 1000:
        raise ValueError(f'Only {len(dataset)} usable observations; investigate data coverage.')
    days = dataset.date.sort_values().unique()
    train_end, validation_end = int(len(days) * .6), int(len(days) * .8)
    dataset['split'] = np.select([dataset.date < days[train_end],
        dataset.date < days[validation_end]], ['train', 'validation'], default='test')
    training = dataset[dataset.split.eq('train')]
    thresholds = training.target_pressure.quantile([1 / 3, 2 / 3]).to_numpy()
    if len(np.unique(thresholds)) != 2:
        raise ValueError('Training quantiles collapse; change the declared target design explicitly.')
    dataset['target_class'] = assign_labels(dataset.target_pressure, thresholds)
    if set(training.assign(target_class=assign_labels(training.target_pressure, thresholds)).target_class) != {0, 1, 2}:
        raise ValueError('Training data must contain all three target classes.')
    quality = {
        'hourly_grid_rows': len(panel), 'analytical_window_rows': int(window.sum()),
        'excluded_silent_block_days': int(panel.loc[~covered, ['blockface_id', 'date']].drop_duplicates().shape[0]),
        'excluded_window_rows_missing_target_or_history': int(window.sum() - len(dataset)),
        'modeling_rows': len(dataset), 'thresholds_training_only': thresholds.tolist(),
        'sampled_blockfaces': int(meters.blockface_id.nunique()),
        'blockfaces_with_any_events': int(events.blockface_id.nunique()),
        'modeled_blockfaces': int(dataset.blockface_id.nunique()),
        'sampled_meters': len(meters), 'meters_with_any_events': int(events.post_id.nunique()),
        'silent_sampled_blockfaces': sorted(set(meters.blockface_id) - set(events.blockface_id)),
        'modeled_weekday_counts': {str(k): int(v) for k, v in dataset.day_of_week.value_counts().sort_index().items()},
        'label_definition': 'NS (new-session) event starts per snapshot eligible single-space meter '
                            'in the target hour, split by training-only terciles. Proxy for paid activity, not occupancy.',
        'analytical_window': 'Published hours 09:00-18:59, Monday-Saturday; excludes Sunday, '
                             'does not verify location-specific charging schedules.',
        'event_counts_by_type': {str(k): int(v) for k, v in events.groupby('meter_event_type').event_count.sum().items()},
        'split_summary': {},
    }
    for name, group in dataset.groupby('split', sort=False):
        quality['split_summary'][name] = {'rows': len(group), 'dates': group.date.nunique(),
            'start': group.target_start.min().isoformat(), 'end': group.target_start.max().isoformat(),
            'class_counts': {str(k): int(v) for k, v in group.target_class.value_counts().sort_index().items()}}
    dataset.to_csv(PROCESSED / 'modeling_data.csv', index=False)
    panel.to_csv(PROCESSED / 'hourly_panel.csv.gz', index=False, compression='gzip')
    (RESULTS / 'data_quality.json').write_text(json.dumps(quality, indent=2))
    supply.to_csv(RESULTS / 'selected_blocks.csv')
    coverage = events.groupby('blockface_id').agg(reporting_meters=('post_id', 'nunique'),
        source_events=('event_count', 'sum'))
    coverage = supply.join(coverage).fillna({'reporting_meters': 0, 'source_events': 0})
    coverage['reporting_meter_fraction'] = coverage.reporting_meters / coverage.supply
    coverage['modeling_rows'] = dataset.groupby('blockface_id').size().reindex(coverage.index, fill_value=0)
    coverage.to_csv(RESULTS / 'block_coverage.csv')
    return dataset, thresholds, quality


def make_pipeline(estimator, numeric=NUMERIC):
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler(), list(numeric)),
        ('location', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', estimator)])


def score(y, prediction, probabilities=None):
    result = {
        'accuracy': accuracy_score(y, prediction),
        'balanced_accuracy': recall_score(y, prediction, labels=np.unique(y), average='macro', zero_division=0),
        'macro_precision': precision_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0),
        'macro_recall': recall_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0),
        'macro_f1': f1_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0),
        'weighted_f1': f1_score(y, prediction, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y, prediction) if len(np.unique(np.r_[y, prediction])) > 1 else 0.,
        'quadratic_kappa': cohen_kappa_score(y, prediction, labels=[0, 1, 2], weights='quadratic')
            if len(np.unique(np.r_[y, prediction])) > 1 else float('nan'),
        'high_activity_recall': recall_score(y, prediction, labels=[2], average='macro', zero_division=0),
    }
    if probabilities is not None:
        probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        result['log_loss'] = log_loss(y, probabilities, labels=[0, 1, 2])
        result['roc_auc_ovr_macro'] = roc_auc_score(y, probabilities, multi_class='ovr', average='macro', labels=[0, 1, 2])
    return {name: float(value) for name, value in result.items()}


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    plt.close()


def plots(dataset, metrics, predictions, best_name, best_model):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    palette = ['#477998', '#EDAE49', '#B85C38']
    counts = dataset.groupby(['split', 'target_class']).size().unstack(fill_value=0).reindex(['train', 'validation', 'test'])
    ax = counts.plot.bar(color=palette, figsize=(8, 4), rot=0)
    ax.set(xlabel='', ylabel='Observations', title='Payment-activity classes by chronological split')
    ax.legend(['Low', 'Medium', 'High'], title='Payment activity')
    save_figure('class_distribution')

    heat = dataset.pivot_table(index='day_of_week', columns='hour', values='target_pressure', aggfunc='mean').reindex(range(6))
    plt.figure(figsize=(10, 4))
    color_map = plt.get_cmap('YlOrRd').copy()
    color_map.set_bad('#DDDDDD')
    plt.imshow(heat.values, aspect='auto', cmap=color_map)
    for row_index, row in enumerate(heat.values):
        if np.isnan(row).all():
            plt.text((len(heat.columns)-1)/2, row_index, 'No usable observations after history filtering',
                     ha='center', va='center', fontsize=9)
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(6), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    plt.xlabel('Published session-start hour (timezone offset unspecified)')
    plt.colorbar(label='New-session starts per eligible meter')
    plt.title('Observed paid activity by weekday and hour')
    save_figure('weekday_hour_heatmap')

    ranked = metrics.sort_values('macro_f1')
    plt.figure(figsize=(10, 5))
    plt.barh(ranked.model, ranked.macro_f1, color=['#477998' if name == best_name else '#9AB3C5' for name in ranked.model])
    plt.xlim(0, 1)
    plt.xlabel('Macro F1 on the chronological test set')
    plt.title('Classical models and forecasting baselines')
    save_figure('model_comparison')

    test = dataset[dataset.split.eq('test')]
    matrix = confusion_matrix(test.target_class, predictions[best_name], labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        plt.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max() / 2 else 'black')
    plt.xticks(range(3), ['Low', 'Medium', 'High'])
    plt.yticks(range(3), ['Low', 'Medium', 'High'])
    plt.xlabel('Predicted activity'); plt.ylabel('Observed activity')
    plt.title(f'{best_name}: test confusion matrix')
    save_figure('confusion_matrix')

    importance = permutation_importance(best_model, test[FEATURES], test.target_class,
        scoring='f1_macro', n_repeats=5, random_state=SEED, n_jobs=1)
    feature_importance = pd.DataFrame({'feature': FEATURES, 'mean': importance.importances_mean,
        'std': importance.importances_std}).sort_values('mean', ascending=False)
    feature_importance.to_csv(RESULTS / 'permutation_importance.csv', index=False)
    top = feature_importance.head(10).iloc[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(top.feature, top['mean'], xerr=top['std'], color='#477998')
    plt.xlabel('Decrease in test macro F1 after permutation')
    plt.title('Feature importance for the validation-selected model')
    save_figure('feature_importance')

    plt.figure(figsize=(9, 5))
    for name, group in dataset.groupby('split', sort=False):
        daily = group.groupby('date').target_pressure.mean()
        plt.plot(daily.index, daily.values, label=name, marker='.', linewidth=1.5)
    plt.ylabel('Mean new-session starts per eligible meter')
    plt.xlabel('Target date'); plt.legend(); plt.xticks(rotation=25)
    plt.title('Data chronology and observed paid activity')
    save_figure('chronological_split')

    blocks = pd.read_csv(RESULTS / 'selected_blocks.csv')
    plt.figure(figsize=(7, 7))
    plt.scatter(blocks.longitude, blocks.latitude, s=blocks.supply * 10, c=blocks.supply, cmap='Blues', edgecolor='#29465B')
    for _, row in blocks.iterrows():
        plt.annotate(str(row.blockface_id), (row.longitude, row.latitude), xytext=(4, 4), textcoords='offset points', fontsize=7)
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.title('Inventory-sampled street blockfaces (no basemap)')
    plt.colorbar(label='Eligible single-space meters')
    save_figure('sampled_locations')


def main():
    for directory in [PROCESSED, RESULTS, FIGURES, ROOT / 'models']:
        directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data, thresholds, quality = build_dataset()
    train = data[data.split.eq('train')]
    validation = data[data.split.eq('validation')]
    test = data[data.split.eq('test')]
    fit_data = data[~data.split.eq('test')]
    candidates = {
        'Logistic Regression': [LogisticRegression(C=c, max_iter=1500, random_state=SEED) for c in [.1, 1, 10]],
        'KNN': [KNeighborsClassifier(n_neighbors=k, weights='distance') for k in [7, 21, 51]],
        'Decision Tree': [DecisionTreeClassifier(max_depth=d, min_samples_leaf=15, random_state=SEED) for d in [4, 8, 12]],
        'Random Forest': [RandomForestClassifier(n_estimators=150, max_depth=d, min_samples_leaf=8,
            n_jobs=2, random_state=SEED) for d in [6, 10, None]],
    }
    tuning, selected = [], {}
    for name, estimators in candidates.items():
        for estimator in estimators:
            model = make_pipeline(estimator)
            model.fit(train[FEATURES], train.target_class)
            value = f1_score(validation.target_class, model.predict(validation[FEATURES]), average='macro')
            tuning.append({'model': name, 'parameters': str(estimator), 'validation_macro_f1': value})
            if name not in selected or value > selected[name][0]:
                selected[name] = (value, clone(model))
        print(name, 'best validation macro F1:', round(selected[name][0], 4), flush=True)
    best_name = max(selected, key=lambda name: selected[name][0])
    pd.DataFrame(tuning).to_csv(RESULTS / 'validation_search.csv', index=False)
    fitted, predictions, metrics, reports = {}, {}, [], {}
    for name, (validation_score, model) in selected.items():
        tick = time.time(); model.fit(fit_data[FEATURES], fit_data.target_class)
        fit_seconds = time.time() - tick
        tick = time.time(); prediction = model.predict(test[FEATURES]); predict_seconds = time.time() - tick
        row = {'model': name, **score(test.target_class, prediction, model.predict_proba(test[FEATURES])),
            'validation_macro_f1': float(validation_score),
            'fit_macro_f1': float(f1_score(fit_data.target_class, model.predict(fit_data[FEATURES]), average='macro')),
            'fit_seconds': fit_seconds, 'test_prediction_seconds': predict_seconds}
        metrics.append(row); predictions[name] = prediction; fitted[name] = model
        reports[name] = classification_report(test.target_class, prediction, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    dummy = DummyClassifier(strategy='most_frequent').fit(fit_data[FEATURES], fit_data.target_class)
    predictions['Majority baseline'] = dummy.predict(test[FEATURES])
    predictions['Persistence baseline'] = assign_labels(test.pressure_lag1, thresholds)
    seasonal = fit_data.groupby(['blockface_id', 'day_of_week', 'hour']).target_pressure.mean()
    seasonal_test = pd.MultiIndex.from_frame(test[['blockface_id', 'day_of_week', 'hour']])
    seasonal_values = seasonal.reindex(seasonal_test).fillna(fit_data.target_pressure.mean()).to_numpy()
    predictions['Seasonal baseline'] = assign_labels(seasonal_values, thresholds)
    for name in ['Majority baseline', 'Persistence baseline', 'Seasonal baseline']:
        metrics.append({'model': name, **score(test.target_class, predictions[name])})
    metrics_frame = pd.DataFrame(metrics).sort_values('macro_f1', ascending=False)
    metrics_frame.to_csv(RESULTS / 'test_metrics.csv', index=False)
    (RESULTS / 'classification_reports.json').write_text(json.dumps(reports, indent=2))
    output = test[['blockface_id', 'target_start', 'date', 'target_pressure', 'target_class']].copy()
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_csv(RESULTS / 'test_predictions.csv', index=False)

    error_records = []
    diagnostic_test = test.copy()
    diagnostic_test['prediction'] = predictions[best_name]
    for dimension in ['blockface_id', 'hour', 'day_of_week']:
        for value, group in diagnostic_test.groupby(dimension):
            error_records.append({'dimension': dimension, 'value': str(value), 'rows': len(group),
                **score(group.target_class, group.prediction)})
    pd.DataFrame(error_records).to_csv(RESULTS / 'error_analysis.csv', index=False)

    # Paired day resampling preserves within-day spatial dependence. Eight test dates
    # cannot establish general seasonal or citywide superiority.
    rng = np.random.default_rng(SEED)
    day_indices = [np.flatnonzero(test.date.to_numpy() == day) for day in test.date.unique()]
    differences = []
    truth = test.target_class.to_numpy()
    for _ in range(1000):
        indices = np.concatenate([day_indices[i] for i in rng.integers(0, len(day_indices), len(day_indices))])
        differences.append(f1_score(truth[indices], predictions[best_name][indices], average='macro') -
                           f1_score(truth[indices], predictions['Seasonal baseline'][indices], average='macro'))
    uncertainty = {'comparison': f'{best_name} minus Seasonal baseline, chronological-test macro F1',
        'point_difference': float(f1_score(truth, predictions[best_name], average='macro') -
                                  f1_score(truth, predictions['Seasonal baseline'], average='macro')),
        'paired_day_bootstrap_95_percentile_interval': np.quantile(differences, [.025, .975]).tolist(),
        'replicates': 1000, 'test_dates': len(day_indices),
        'caveat': 'Descriptive interval from few test dates; adjacent dates can remain dependent. '
                  'Not evidence of citywide or seasonal generalization.'}
    (RESULTS / 'uncertainty.json').write_text(json.dumps(uncertainty, indent=2))

    # Ablations use the frozen hyperparameters and the same chronological partitions.
    ablations = []
    for label, numeric in [('Calendar and location', CALENDAR + ['latitude', 'longitude']),
                            ('Calendar, location and supply', CALENDAR + STATIC),
                            ('Calendar, location, supply and history', NUMERIC)]:
        estimator = clone(fitted[best_name].named_steps['model'])
        model = make_pipeline(estimator, numeric)
        model.fit(fit_data[FEATURES], fit_data.target_class)
        ablations.append({'features': label, **score(test.target_class, model.predict(test[FEATURES]))})
    pd.DataFrame(ablations).to_csv(RESULTS / 'ablation_metrics.csv', index=False)

    # Auxiliary experiment only. It does not select the final model or replace the future test.
    random_train, random_test = train_test_split(data, test_size=.2, random_state=SEED, stratify=data.target_class)
    random_model = clone(fitted[best_name]).fit(random_train[FEATURES], random_train.target_class)
    diagnostic = {'warning': 'Separate shuffled-split diagnostic, with different test observations. '
                  'Labels retain the chronological-training thresholds. No causal attribution of any score difference.',
                  'model': best_name, 'random_test_rows': len(random_test),
                  'random_split': score(random_test.target_class, random_model.predict(random_test[FEATURES])),
                  'chronological_test': score(test.target_class, predictions[best_name])}
    (RESULTS / 'split_diagnostic.json').write_text(json.dumps(diagnostic, indent=2))

    best = fitted[best_name]
    bundle = {'pipeline': best, 'thresholds': thresholds.tolist(), 'features': FEATURES,
        'class_names': CLASS_NAMES, 'selected_model': best_name,
        'forecast_definition': 'At the target-hour boundary, predict NS starts per eligible meter '
                               'during the coming hour. Recent payment aggregates are assumed available immediately.',
        'limitations': 'Paid activity proxy, not ground-truth occupancy. Snapshot supply and unverified '
                      'timestamp offset. One-hour rolling evaluation assumes recent observations are available.',
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__}}
    model_path = ROOT / 'models/parking_activity_model.joblib'
    joblib.dump(bundle, model_path, compress=3)
    example = test.iloc[[0]][FEATURES + ['target_start']].copy()
    example['target_start'] = example.target_start.dt.strftime('%Y-%m-%dT%H:%M:%S')
    example.to_json(ROOT / 'models/example_input.json', orient='records', indent=2)
    # Verify persistence and feature schema against a real held-out observation.
    restored = joblib.load(model_path)
    assert np.array_equal(best.predict(example[FEATURES]), restored['pipeline'].predict(example[FEATURES]))
    plots(data, metrics_frame, predictions, best_name, best)

    summary = {'selected_model_by_validation': best_name,
        'selected_validation_macro_f1': float(selected[best_name][0]),
        'selected_test_metrics': score(test.target_class, predictions[best_name], best.predict_proba(test[FEATURES])),
        'best_baseline_test_macro_f1': float(metrics_frame[metrics_frame.model.str.contains('baseline')].macro_f1.max()),
        'model_bytes': model_path.stat().st_size, 'total_runtime_seconds': time.time() - started,
        'data_quality': quality, 'versions': bundle['versions'],
        'protocol': 'Training-only target terciles; chronological date split 60/20/20; '
                    'three configurations per algorithm ranked on validation macro F1; '
                    'refit each selected configuration on train+validation; test once. '
                    'Ablation and split diagnostic are descriptive post-selection analyses.',
        'feature_availability': 'All payment-derived features use completed hours earlier than target_start. '
                                'Calendar features describe the coming hour. Inventory is a fixed later snapshot.',
        'inference_latency_assumption': 'No measured API ingestion latency; immediate hourly aggregate '
                                       'availability is an experimental assumption, not a deployment claim.'}
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(metrics_frame[['model', 'accuracy', 'macro_f1', 'balanced_accuracy']].to_string(index=False))
    print('Selected model:', best_name, '| Saved model bytes:', model_path.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
