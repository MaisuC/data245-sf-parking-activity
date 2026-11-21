"""Quick demo using a saved model and actual held-out features (no retraining)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'models/example_input.json')
    args = parser.parse_args()
    # Load only this project's trusted model file. Pickle-based files can execute code.
    bundle = joblib.load(ROOT / 'models/parking_activity_model.joblib')
    frame = pd.read_json(args.input, dtype={'blockface_id': str})
    missing = set(bundle['features']) - set(frame.columns)
    if missing:
        parser.error(f'Missing model features: {sorted(missing)}')
    if frame[bundle['features']].isna().any().any():
        parser.error('Model features must not contain missing values.')
    prediction = bundle['pipeline'].predict(frame[bundle['features']])
    probabilities = bundle['pipeline'].predict_proba(frame[bundle['features']])
    for i, label in enumerate(prediction):
        print(json.dumps({'row': i, 'model': bundle['selected_model'],
            'blockface_id': str(frame.iloc[i].blockface_id),
            'target_start': str(frame.iloc[i].get('target_start', 'Not provided')),
            'predicted_class': int(label), 'description': bundle['class_names'][label],
            'probabilities': {name: round(float(p), 4) for name, p in zip(bundle['class_names'], probabilities[i])},
            'interpretation': 'Forecast of meter payment activity, not an observed free-space count.'}, indent=2))


if __name__ == '__main__':
    main()
