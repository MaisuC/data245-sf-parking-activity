"""Checks for the Downtown arrival-time pipeline on small synthetic sessions (no network)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.downtown_pipeline import (FEATURE_SETS, add_seasonal, assign_splits, block_panel,  # noqa: E402
                                   build_examples, load_sessions, status_grid)
from src.downtown_supply import block_table, select_downtown  # noqa: E402

DAY = pd.Timestamp('2026-09-15')


def sessions_frame(rows):
    return pd.DataFrame(rows, columns=['post_id', 'session_start_dt', 'session_end_dt', 'meter_event_type'])


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.meters = pd.DataFrame({'post_id': ['p1', 'p2'], 'blockface_id': ['b1', 'b1']})

    def at(self, times, hhmm):
        return list(times).index(DAY + pd.Timedelta(hhmm))

    def test_add_time_extends_only_after_it_is_paid(self):
        raw = sessions_frame([
            ['p1', f'{DAY.date()}T10:00:00', f'{DAY.date()}T10:40:00', 'NS'],
            ['p1', f'{DAY.date()}T10:30:00', f'{DAY.date()}T11:00:00', 'AT']])
        s = load_sessions(raw, self.meters)
        times, q, paid, until = status_grid(s, self.meters, [DAY])
        k = self.at(times, '10:25:00')
        self.assertTrue(paid[0, k])
        self.assertEqual(until[0, k], q[k] + 15)        # known end is still 10:40 at 10:25
        k2 = self.at(times, '10:45:00')
        self.assertTrue(paid[0, k2])                     # extension made at 10:30 applies
        self.assertFalse(paid[0, self.at(times, '11:00:00')])
        self.assertFalse(paid[1].any())                  # p2 never paid

    def test_expiring_counts_known_ends_within_15_minutes(self):
        raw = sessions_frame([
            ['p1', f'{DAY.date()}T10:00:00', f'{DAY.date()}T10:40:00', 'NS'],
            ['p2', f'{DAY.date()}T10:00:00', f'{DAY.date()}T12:00:00', 'NS']])
        s = load_sessions(raw, self.meters)
        blocks = pd.DataFrame({'blockface_id': ['b1'], 'supply': [2], 'latitude': [37.79], 'longitude': [-122.4]})
        panel = block_panel(s, self.meters, blocks, [DAY])
        k = list(panel['times']).index(DAY + pd.Timedelta('10:25:00'))
        self.assertEqual(panel['paid'][0, k], 2)
        self.assertEqual(panel['expiring'][0, k], 1)
        self.assertEqual(panel['starts15'][0, list(panel['times']).index(DAY + pd.Timedelta('10:10:00'))], 2)


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        days = pd.date_range('2026-09-01', '2026-10-10', freq='D')
        posts = [f'p{i}' for i in range(12)]
        meters = pd.DataFrame({'post_id': posts, 'blockface_id': ['b1'] * 6 + ['b2'] * 6})
        rows = []
        for d in days:
            for p in posts:
                t = d + pd.Timedelta(hours=8)
                while t < d + pd.Timedelta(hours=18):
                    t += pd.Timedelta(minutes=int(rng.integers(5, 60)))
                    end = t + pd.Timedelta(minutes=int(rng.integers(10, 90)))
                    rows.append([p, t.isoformat(), end.isoformat(), 'NS'])
                    t = end
        cls.meters = meters
        cls.blocks = pd.DataFrame({'blockface_id': ['b1', 'b2'], 'supply': [6, 6],
                                   'latitude': [37.79, 37.791], 'longitude': [-122.40, -122.40]})
        cls.sessions = load_sessions(sessions_frame(rows), meters)
        cls.panel = block_panel(cls.sessions, meters, cls.blocks, days)
        cls.h = 10
        cls.data = add_seasonal(assign_splits(build_examples(cls.panel, cls.blocks, cls.h)))

    def test_target_is_future_and_features_are_past(self):
        times = list(self.panel['times'])
        for _, r in self.data.sample(40, random_state=1).iterrows():
            j = self.panel['blocks'].index(r.blockface_id)
            k = times.index(r.decision_time)
            self.assertEqual(r.arrival_time, r.decision_time + pd.Timedelta(minutes=self.h))
            self.assertEqual(r.unpaid_future, 6 - min(self.panel['paid'][j, k + self.h // 5], 6))
            self.assertAlmostEqual(r.occ_now, min(self.panel['paid'][j, k], 6) / 6)

    def test_feature_sets_exclude_outcomes(self):
        for cols in FEATURE_SETS.values():
            for forbidden in ['y', 'unpaid_future', 'arrival_time']:
                self.assertNotIn(forbidden, cols)
        self.assertTrue(set(FEATURE_SETS['B']) < set(FEATURE_SETS['C']))

    def test_splits_are_in_date_order(self):
        d = self.data
        self.assertLess(d[d.split.eq('train')].date.max(), d[d.split.eq('calibration')].date.min())
        self.assertLess(d[d.split.eq('calibration')].date.max(), d[d.split.eq('test')].date.min())
        self.assertTrue((d.decision_time.dt.dayofweek != 6).all())

    def test_seasonal_rate_uses_training_days_only(self):
        flipped = self.data.copy()
        test = flipped.split.ne('train')
        flipped.loc[test, 'y'] = 1 - flipped.loc[test, 'y']
        np.testing.assert_allclose(add_seasonal(flipped).seasonal_rate, self.data.seasonal_rate)


class SupplyTests(unittest.TestCase):
    def test_selection_and_choice_sets(self):
        base = {'analysis_neighborhood': 'Inner Richmond', 'on_offstreet_type': 'ON', 'meter_vendor': 'MacKay',
                'active_meter_flag': 'M', 'meter_type': 'SS', 'jurisdiction': 'SFMTA', 'street_name': 'X'}
        rows = [dict(base, post_id='a', blockface_id='1', cap_color='Grey', latitude='37.790', longitude='-122.4'),
                dict(base, post_id='b', blockface_id='2', cap_color='Green', latitude='37.791', longitude='-122.4'),
                dict(base, post_id='c', blockface_id='3', cap_color='Yellow', latitude='37.790', longitude='-122.4'),
                dict(base, post_id='d', blockface_id='4', cap_color='Grey', latitude='37.800', longitude='-122.4'),
                dict(base, post_id='e', blockface_id='5', cap_color='Grey', latitude='37.79', longitude='-122.4',
                     analysis_neighborhood='Mission'),
                dict(base, post_id='f', blockface_id='6', cap_color='Grey', latitude='37.79', longitude='-122.4',
                     meter_vendor='IPS')]   # IPS meters report no payments, so they are left out
        meters = select_downtown(pd.DataFrame(rows))
        self.assertEqual(sorted(meters.post_id), ['a', 'b', 'd'])
        blocks = block_table(meters).set_index('blockface_id')
        self.assertEqual(blocks.loc['1', 'blocks_within_200m'], 1)   # block 2 (~111 m)
        self.assertEqual(blocks.loc['4', 'blocks_within_400m'], 0)   # ~1.1 km away


if __name__ == '__main__':
    unittest.main()
