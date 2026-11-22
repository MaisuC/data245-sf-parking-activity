"""Downtown supply layer: which spaces an ordinary driver may use, and nearby blocks.

Adapted from AlaaTarekA/sf-parking-availability (src/sf_supply.py) for this repo.
Uses the inventory cached by src/download_data.py (data/raw/meters.json).

Selection: Financial District / South Beach, active (M/T) on-street single-space
SFMTA meters with a grey (general) or green (short-term) cap. Each blockface is a
block; its choice set is every other block with usable spaces within 200 m / 400 m.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.download_data import get_json  # noqa: E402

RAW, PROCESSED = ROOT / 'data' / 'raw', ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results' / 'downtown'
NEIGHBORHOOD = 'Financial District/South Beach'
CAP_USE = {'Grey': 'general', 'Green': 'short-term', 'Yellow': 'commercial loading',
           'Red': 'truck loading', 'Black': 'motorcycle'}
DRIVER_CAPS = ['Grey', 'Green']
RADII_M = (200, 400)


def load_inventory():
    path = RAW / 'meters.json'
    if path.exists():
        return json.loads(path.read_text())
    inventory, _ = get_json('/resource/8vzz-qzz9.json', {'$limit': 50000})
    if len(inventory) >= 50000:
        raise RuntimeError('Inventory limit reached; pagination is required.')
    path.write_text(json.dumps(inventory))
    return inventory


def area_profile(frame):
    """Counts for the whole Downtown area, before keeping driver-usable meters."""
    area = frame[frame.analysis_neighborhood.eq(NEIGHBORHOOD) & frame.on_offstreet_type.eq('ON') &
                 frame.active_meter_flag.isin(['M', 'T'])]
    uses = area.cap_color.map(CAP_USE).fillna('other').value_counts()
    return {'active_on_street_meters': len(area), 'blocks': int(area.blockface_id.nunique()),
            'meters_by_use': {k: int(v) for k, v in uses.items()},
            'driver_usable_share': round(float(area.cap_color.isin(DRIVER_CAPS).mean()), 3)}


def select_downtown(frame):
    keep = (frame.analysis_neighborhood.eq(NEIGHBORHOOD) & frame.on_offstreet_type.eq('ON') &
            frame.active_meter_flag.isin(['M', 'T']) & frame.meter_type.eq('SS') &
            frame.cap_color.isin(DRIVER_CAPS) & frame.jurisdiction.eq('SFMTA'))
    meters = frame.loc[keep].drop_duplicates('post_id').copy()
    for column in ['blockface_id', 'latitude', 'longitude']:
        meters[column] = pd.to_numeric(meters[column], errors='coerce')
    meters = meters.dropna(subset=['blockface_id', 'latitude', 'longitude'])
    meters['blockface_id'] = meters.blockface_id.astype(int).astype(str)
    return meters.sort_values('post_id')


def distances(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 +
         np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def block_table(meters):
    blocks = meters.groupby('blockface_id').agg(
        supply=('post_id', 'nunique'), latitude=('latitude', 'mean'), longitude=('longitude', 'mean'),
        street_name=('street_name', 'first')).reset_index()
    dist = distances(blocks.latitude.to_numpy(), blocks.longitude.to_numpy())
    np.fill_diagonal(dist, np.inf)
    for radius in RADII_M:
        near = dist <= radius
        blocks[f'blocks_within_{radius}m'] = near.sum(1)
        blocks[f'spaces_within_{radius}m'] = near @ blocks.supply.to_numpy()
    return blocks


def main():
    frame = pd.DataFrame(load_inventory())
    meters = select_downtown(frame)
    if meters.empty:
        raise RuntimeError('No Downtown meters selected; check inventory fields.')
    blocks = block_table(meters)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    meters.to_csv(RAW / 'downtown_meters.csv', index=False)
    blocks.to_csv(PROCESSED / 'downtown_blocks.csv', index=False)
    summary = {'area': area_profile(frame), 'modeled_meters': len(meters),
               'modeled_blocks': len(blocks), 'caps': meters.cap_color.value_counts().to_dict(),
               **{f'median_blocks_within_{r}m': float(blocks[f'blocks_within_{r}m'].median()) for r in RADII_M},
               **{f'median_spaces_within_{r}m': float(blocks[f'spaces_within_{r}m'].median()) for r in RADII_M},
               'selection': 'Financial District/South Beach; active M/T, ON, SS, SFMTA; Grey or Green cap.'}
    (RESULTS / 'supply_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
