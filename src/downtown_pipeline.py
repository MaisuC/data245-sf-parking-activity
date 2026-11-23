"""Turn Downtown payment sessions into arrival-time examples without using the future.

Paid status of a meter at time q uses only payment records that started at or
before q: paid_until(q) = latest session end among those records, and the meter
is paid at q when paid_until(q) > q. Add-time records extend paid_until only from
the moment they are made, so later extensions never leak into earlier features.

Target (horizon h): the block has at least one meter WITHOUT an active payment at
t + h. This is "a space with no active payment", not verified free parking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID_MIN = 5                           # status grid resolution, minutes
STEP_MIN = 10                          # one decision every 10 minutes
FIRST_DECISION, LAST_ARRIVAL = 9 * 60, 18 * 60   # 09:00 decisions, arrivals by 18:00
HORIZONS = (5, 10, 15)
CALENDAR = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_saturday']
STATIC = ['supply', 'latitude', 'longitude']
HISTORY = ['seasonal_rate', 'occ_lag1d', 'occ_lag7d']
LIVE = ['occ_now', 'unpaid_now', 'any_unpaid_now', 'occ_lag5', 'occ_lag15', 'starts_last15']
PAID_TIME = ['expiring_15', 'expiring_share_15']
NEXT_HOUR = ['p1_low', 'p1_high']    # Part 1's next-hour forecast (see src/hourly_forecast.py)
FEATURE_SETS = {'A': CALENDAR + STATIC + HISTORY,
                'B': CALENDAR + STATIC + HISTORY + LIVE,
                'C': CALENDAR + STATIC + HISTORY + LIVE + PAID_TIME,
                'D': CALENDAR + STATIC + HISTORY + LIVE + PAID_TIME + NEXT_HOUR}


def load_sessions(sessions, meters):
    s = sessions.copy()
    s['start'] = pd.to_datetime(s.session_start_dt, errors='coerce')
    s['end'] = pd.to_datetime(s.session_end_dt, errors='coerce')
    s = s.dropna(subset=['start', 'end'])
    s = s[(s.end > s.start) & (s.end - s.start <= pd.Timedelta(hours=12))]
    s = s.merge(meters[['post_id', 'blockface_id']], on='post_id', validate='many_to_one')
    return s.sort_values(['post_id', 'start'])


def status_grid(sessions, meters, days):
    """Per meter and 5-minute grid time: paid flag and paid_until, using records started by then."""
    times = pd.DatetimeIndex([d + pd.Timedelta(minutes=m) for d in days for m in range(0, 24 * 60, GRID_MIN)])
    q = times.to_numpy().astype('datetime64[m]').astype(np.int64)
    posts = meters.post_id.tolist()
    paid = np.zeros((len(posts), len(q)), dtype=bool)
    until = np.full((len(posts), len(q)), -1, dtype=np.int64)
    grouped = dict(tuple(sessions.groupby('post_id')))
    for i, post in enumerate(posts):
        g = grouped.get(post)
        if g is None:
            continue
        starts = g.start.to_numpy().astype('datetime64[m]').astype(np.int64)
        ends = np.maximum.accumulate(g.end.to_numpy().astype('datetime64[m]').astype(np.int64))
        idx = np.searchsorted(starts, q, side='right') - 1
        known = idx >= 0
        until[i, known] = ends[idx[known]]
        paid[i] = known & (until[i] > q)
    return times, q, paid, until


def block_panel(sessions, meters, blocks, days):
    times, q, paid, until = status_grid(sessions, meters, days)
    block_of = meters.blockface_id.to_numpy()
    ids = blocks.blockface_id.tolist()
    member = np.stack([block_of == b for b in ids]).astype(np.int32)        # blocks x meters
    paid_count = member @ paid.astype(np.int32)
    # Paid sessions known at q that run out within the next 15 minutes (RQ2).
    expiring = member @ ((until > q) & (until <= q + 15)).astype(np.int32)
    ns = sessions[sessions.meter_event_type.eq('NS')]
    starts_last15 = np.zeros_like(paid_count)
    for j, b in enumerate(ids):
        st = np.sort(ns.loc[ns.blockface_id.eq(b), 'start'].to_numpy().astype('datetime64[m]').astype(np.int64))
        starts_last15[j] = np.searchsorted(st, q, side='right') - np.searchsorted(st, q - 15, side='right')
    covered = sessions.assign(day=sessions.start.dt.normalize()).groupby(['blockface_id', 'day']).size()
    return {'times': times, 'blocks': ids, 'paid': paid_count, 'expiring': expiring,
            'starts15': starts_last15, 'covered': set(covered.index)}


def build_examples(panel, blocks, h):
    times, paid = panel['times'], panel['paid']
    supply = blocks.set_index('blockface_id').loc[panel['blocks'], 'supply'].to_numpy()
    occ = np.minimum(paid, supply[:, None]) / supply[:, None]
    per_day = 24 * 60 // GRID_MIN
    step, lag5, lag15, hh = STEP_MIN // GRID_MIN, 5 // GRID_MIN, 15 // GRID_MIN, h // GRID_MIN
    rows = []
    minute = times.hour * 60 + times.minute
    for k in range(len(times)):
        m = minute[k]
        if m < FIRST_DECISION or m + h > LAST_ARRIVAL or (m - FIRST_DECISION) % STEP_MIN:
            continue
        t = times[k]
        if t.dayofweek == 6:
            continue
        day = t.normalize()
        ok = np.array([(b, day) in panel['covered'] for b in panel['blocks']])
        if k - 7 * per_day >= 0:
            ok &= np.array([(b, day - pd.Timedelta(days=7)) in panel['covered'] for b in panel['blocks']])
        else:
            continue
        if not ok.any():
            continue
        future_unpaid = supply - np.minimum(paid[:, k + hh], supply)
        rows.append(pd.DataFrame({
            'blockface_id': np.array(panel['blocks'])[ok], 'decision_time': t,
            'arrival_time': t + pd.Timedelta(minutes=h), 'date': day, 'supply': supply[ok],
            'occ_now': occ[ok, k], 'unpaid_now': (supply - np.minimum(paid[:, k], supply))[ok],
            'occ_lag5': occ[ok, k - lag5], 'occ_lag15': occ[ok, k - lag15],
            'occ_lag1d': occ[ok, k - per_day], 'occ_lag7d': occ[ok, k - 7 * per_day],
            'starts_last15': panel['starts15'][ok, k], 'expiring_15': panel['expiring'][ok, k],
            'unpaid_future': future_unpaid[ok]}))
    data = pd.concat(rows, ignore_index=True)
    data['any_unpaid_now'] = (data.unpaid_now >= 1).astype(int)
    data['expiring_share_15'] = data.expiring_15 / data.supply
    data['y'] = (data.unpaid_future >= 1).astype(int)
    hour = data.decision_time.dt.hour + data.decision_time.dt.minute / 60
    dow = data.decision_time.dt.dayofweek
    data['hour'] = data.decision_time.dt.hour
    data['dow'] = dow
    data['hour_sin'], data['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    data['dow_sin'], data['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    data['is_saturday'] = dow.eq(5).astype(int)
    loc = blocks.set_index('blockface_id')[['latitude', 'longitude']]
    data = data.join(loc, on='blockface_id')
    return data


def assign_splits(data):
    """Same rule as src/train.py: first 60% of dates train, next 20% calibration, last 20% test."""
    days = np.sort(data.date.unique())
    a, b = int(len(days) * .6), int(len(days) * .8)
    data['split'] = np.select([data.date < days[a], data.date < days[b]], ['train', 'calibration'], 'test')
    return data


def add_seasonal(data):
    train = data[data.split.eq('train')]
    rate = train.groupby(['blockface_id', 'dow', 'hour']).y.mean()
    block_rate = train.groupby('blockface_id').y.mean()
    key = pd.MultiIndex.from_frame(data[['blockface_id', 'dow', 'hour']])
    data['seasonal_rate'] = rate.reindex(key).to_numpy()
    data['seasonal_rate'] = data.seasonal_rate.fillna(data.blockface_id.map(block_rate)).fillna(train.y.mean())
    return data
