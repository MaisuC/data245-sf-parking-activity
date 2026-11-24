"""Download session-level payments for the Downtown meters (start and end of each paid session).

Run src/downtown_supply.py first. Reuses get_json from src/download_data.py.
Output: data/raw/downtown_sessions.csv.gz and data/raw/downtown_manifest.json (not committed: large).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.download_data import BASE, get_json  # noqa: E402

RAW = ROOT / 'data' / 'raw'
PAGE = 50000
FIELDS = 'post_id,session_start_dt,session_end_dt,meter_event_type'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-09-01')
    parser.add_argument('--end', default='2026-11-01', help='Exclusive end date')
    args = parser.parse_args()
    meters = pd.read_csv(RAW / 'downtown_meters.csv', dtype={'post_id': str})
    ids = meters.post_id.tolist()
    chunks = [ids[i:i + 40] for i in range(0, len(ids), 40)]
    cache_dir = RAW / 'queries_downtown'
    cache_dir.mkdir(parents=True, exist_ok=True)

    def download_chunk(item):
        index, posts = item
        where = (f"session_start_dt >= '{args.start}T00:00:00' AND "
                 f"session_start_dt < '{args.end}T00:00:00' AND post_id in (" +
                 ','.join("'" + value.replace("'", "''") + "'" for value in posts) + ')')
        rows, offset = [], 0
        while True:
            params = {'$select': FIELDS, '$where': where, '$order': 'post_id,session_start_dt',
                      '$limit': PAGE, '$offset': offset}
            key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
            cache = cache_dir / f'{index:03d}_{offset:07d}_{key}.json'
            if cache.exists():
                page = json.loads(cache.read_text())['rows']
            else:
                page, url = get_json('/resource/imvp-dq3v.json', params)
                cache.write_text(json.dumps({'url': url, 'rows': page,
                    'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}))
            rows.extend(page)
            if len(page) < PAGE:
                break
            offset += PAGE
        print(f'Chunk {index + 1}/{len(chunks)}: {len(rows):,} sessions', flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(download_chunk, enumerate(chunks)))
    sessions = pd.DataFrame([row for part in parts for row in part])
    if sessions.empty:
        raise RuntimeError('No sessions returned. Check date coverage.')
    sessions.to_csv(RAW / 'downtown_sessions.csv.gz', index=False, compression='gzip')
    manifest = {'start_inclusive': args.start, 'end_exclusive': args.end, 'meters': len(meters),
                'blocks': int(meters.blockface_id.nunique()), 'sessions': len(sessions),
                'event_types': sessions.meter_event_type.value_counts().to_dict(),
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'time_basis': 'Published wall-clock session times; no timezone conversion.',
                'source': BASE + '/d/imvp-dq3v'}
    (RAW / 'downtown_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
