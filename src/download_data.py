"""Download public DataSF observations; never generate synthetic training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
BASE = 'https://data.sfgov.org'


def get_json(path, params=None):
    for attempt in range(4):
        try:
            response = requests.get(BASE + path, params=params, timeout=(20, 180))
            response.raise_for_status()
            return response.json(), response.url
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def select_meters(meters, blocks=16):
    frame = pd.DataFrame(meters)
    keep = (frame.active_meter_flag.isin(['M', 'T']) &
            frame.on_offstreet_type.eq('ON') & frame.meter_type.eq('SS') &
            frame.cap_color.eq('Grey') & frame.jurisdiction.eq('SFMTA'))
    frame = frame.loc[keep].drop_duplicates('post_id').copy()
    for column in ['blockface_id', 'latitude', 'longitude']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['blockface_id', 'latitude', 'longitude',
                                 'analysis_neighborhood'])
    frame['blockface_id'] = frame.blockface_id.astype(int).astype(str)
    supply = frame.groupby('blockface_id').agg(
        supply=('post_id', 'nunique'), neighborhood=('analysis_neighborhood', 'first'))
    eligible = supply[supply.supply.between(12, 35)].reset_index()
    # Inventory-only sampling: selection does not inspect future payment outcomes.
    candidates = eligible.sort_values('blockface_id').groupby(
        'neighborhood', group_keys=False).sample(n=1, random_state=245)
    chosen = candidates.sample(n=min(blocks, len(candidates)), random_state=245)
    return frame[frame.blockface_id.isin(chosen.blockface_id)].sort_values('post_id')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-09-01')
    parser.add_argument('--end', default='2026-11-01', help='Exclusive end date')
    parser.add_argument('--blocks', type=int, default=16)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    inventory_path = RAW / 'meters.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        inventory, _ = get_json('/resource/8vzz-qzz9.json', {'$limit': 50000})
        inventory_path.write_text(json.dumps(inventory))
    if len(inventory) >= 50000:
        raise RuntimeError('Inventory limit reached; pagination is required.')
    selected = select_meters(inventory, args.blocks)
    selected.to_csv(RAW / 'selected_meters.csv', index=False)
    ids = selected.post_id.tolist()
    chunks = [ids[i:i + 40] for i in range(0, len(ids), 40)]
    query_dir = RAW / 'queries'
    query_dir.mkdir(exist_ok=True)

    def download_chunk(item):
        index, posts = item
        where = (f"session_start_dt >= '{args.start}T00:00:00' AND "
                 f"session_start_dt < '{args.end}T00:00:00' AND post_id in (" +
                 ','.join("'" + value.replace("'", "''") + "'" for value in posts) + ')')
        params = {
            '$select': 'post_id,date_trunc_ymd(session_start_dt) AS day,'
                       'date_extract_hh(session_start_dt) AS hour,'
                       'meter_event_type,count(*) AS event_count',
            '$where': where,
            '$group': 'post_id,day,hour,meter_event_type',
            '$order': 'post_id,day,hour,meter_event_type', '$limit': 50000,
        }
        signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        cache = query_dir / f'{index:02d}_{signature}.json'
        if cache.exists():
            record = json.loads(cache.read_text())
        else:
            rows, url = get_json('/resource/imvp-dq3v.json', params)
            if len(rows) >= 50000:
                raise RuntimeError('Query limit reached; reduce chunk size.')
            record = {'url': url, 'parameters': params, 'rows': rows,
                      'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
            cache.write_text(json.dumps(record))
        print(f'Chunk {index + 1}/{len(chunks)}: {len(record["rows"]):,} aggregate rows', flush=True)
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download_chunk, enumerate(chunks)))
    rows = [row for record in records for row in record['rows']]
    if not rows:
        raise RuntimeError('No observations returned. Check date coverage.')
    events = pd.DataFrame(rows)
    events['event_count'] = pd.to_numeric(events.event_count)
    events.to_csv(RAW / 'hourly_meter_events.csv', index=False)
    manifest = {
        'start_inclusive': args.start, 'end_exclusive': args.end,
        'inventory_rows': len(inventory), 'selected_meters': len(selected),
        'selected_blockfaces': int(selected.blockface_id.nunique()),
        'aggregate_rows': len(events), 'source_event_count': int(events.event_count.sum()),
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'inventory_as_of_values': sorted(selected.data_as_of.dropna().unique().tolist()),
        'selection': 'Seed 245; one eligible blockface per neighborhood, then 16 neighborhoods; '
                     'active M/T, ON, SS, Grey, SFMTA; 12-35 single-space meters per blockface.',
        'time_basis': 'Published session_start_dt wall-clock values; the API provides no timezone offset. '
                      'No timezone conversion or claim of verified local-time alignment.',
        'sources': {'transactions': BASE + '/d/imvp-dq3v', 'inventory': BASE + '/d/8vzz-qzz9'},
        'queries': [{key: record[key] for key in record if key != 'rows'} for record in records],
    }
    (RAW / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != 'queries'}, indent=2))


if __name__ == '__main__':
    main()
