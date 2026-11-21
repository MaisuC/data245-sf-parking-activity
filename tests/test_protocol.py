"""Check temporal provenance, honest labels, and saved-artifact consistency."""
import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import FEATURES, assign_labels


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / 'data/processed/modeling_data.csv',
            dtype={'blockface_id': str}, parse_dates=['target_start', 'latest_source_hour'])
        cls.panel = pd.read_csv(ROOT / 'data/processed/hourly_panel.csv.gz',
            dtype={'blockface_id': str}, parse_dates=['target_start'])
        cls.quality = json.loads((ROOT / 'results/data_quality.json').read_text())

    def test_future_dates_are_held_out(self):
        train = self.data[self.data.split.eq('train')]
        val = self.data[self.data.split.eq('validation')]
        test = self.data[self.data.split.eq('test')]
        self.assertLess(train.target_start.max(), val.target_start.min())
        self.assertLess(val.target_start.max(), test.target_start.min())
        self.assertFalse(self.data.duplicated(['blockface_id', 'target_start']).any())

    def test_all_history_precedes_the_target_hour(self):
        self.assertTrue((self.data.latest_source_hour < self.data.target_start).all())
        panel = self.panel.sort_values(['blockface_id', 'target_start'])
        for lag in [1, 2, 24, 168]:
            expected = panel.groupby('blockface_id').target_pressure.shift(lag)
            np.testing.assert_allclose(panel[f'pressure_lag{lag}'], expected, equal_nan=True)
        for forbidden in ['target_pressure', 'target_class', 'new_session_count', 'all_events']:
            self.assertNotIn(forbidden, FEATURES)

    def test_thresholds_use_training_data_only(self):
        train = self.data[self.data.split.eq('train')]
        expected = train.target_pressure.quantile([1/3, 2/3]).to_numpy()
        np.testing.assert_allclose(expected, self.quality['thresholds_training_only'])
        # CSV floats can round at a boundary. Use tolerance before testing labels.
        values = self.data.target_pressure.to_numpy().copy()
        for boundary in expected:
            values[np.isclose(values, boundary, atol=1e-12, rtol=0)] = boundary
        np.testing.assert_array_equal(assign_labels(values, expected), self.data.target_class)

    def test_saved_model_reproduces_published_test_predictions(self):
        bundle = joblib.load(ROOT / 'models/parking_activity_model.joblib')
        predictions = pd.read_csv(ROOT / 'results/test_predictions.csv')
        test = self.data[self.data.split.eq('test')]
        actual = bundle['pipeline'].predict(test[bundle['features']])
        np.testing.assert_array_equal(actual, predictions[bundle['selected_model']])

    def test_event_totals_match_download_manifest(self):
        manifest = json.loads((ROOT / 'data/raw/download_manifest.json').read_text())
        events = pd.read_csv(ROOT / 'data/raw/hourly_meter_events.csv')
        self.assertEqual(int(events.event_count.sum()), manifest['source_event_count'])
        self.assertEqual(sum(self.quality['event_counts_by_type'].values()), manifest['source_event_count'])
        selected = pd.read_csv(ROOT / 'data/raw/selected_meters.csv')
        self.assertTrue(set(events.post_id).issubset(set(selected.post_id)))
        self.assertFalse(selected.post_id.duplicated().any())


if __name__ == '__main__':
    unittest.main()
