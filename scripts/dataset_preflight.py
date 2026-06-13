from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model_quality.ingestion.load_cfpb import (  # noqa: E402
    NORMALIZED_REQUIRED,
    discover_cfpb_path,
    load_cfpb,
)


def _disk_free_bytes(path: Path) -> int:
    probe = path if path.exists() else path.parent
    return int(shutil.disk_usage(probe).free)


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate a local CFPB dataset before a full pipeline run.')
    parser.add_argument('--cfpb-path', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--archive-cache-dir', type=str, default=None)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--sampling-strategy', choices=['auto', 'duckdb', 'reservoir', 'head'], default='head')
    parser.add_argument('--output', type=str, default='outputs/dataset_preflight.json')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = Path(args.cfpb_path) if args.cfpb_path else discover_cfpb_path(data_dir / 'raw')
    started = time.perf_counter()
    result: dict = {
        'dataset_path': str(dataset.resolve()),
        'dataset_name': dataset.name,
        'dataset_size_bytes': int(dataset.stat().st_size),
        'extension_supported': dataset.name.lower().endswith(('.csv', '.csv.gz', '.7z')),
        'archive_cache_dir': str(Path(args.archive_cache_dir).resolve()) if args.archive_cache_dir else None,
        'disk_free_bytes_near_dataset': _disk_free_bytes(dataset),
    }

    try:
        sample = load_cfpb(
            dataset,
            sample_size=max(20, int(args.sample_size)),
            random_state=20260612,
            sampling_strategy=args.sampling_strategy,
            archive_cache_dir=args.archive_cache_dir,
        )
        source_present = set(sample.attrs.get('source_required_columns_present', []))
        result.update({
            'passed': True,
            'usable_sample_rows': int(len(sample)),
            'required_columns': NORMALIZED_REQUIRED,
            'required_columns_present': sorted(source_present),
            'required_columns_missing': sorted(set(NORMALIZED_REQUIRED) - source_present),
            'effective_sampling_strategy': sample.attrs.get('effective_sampling_strategy'),
            'date_min': str(sample['date_received'].min()),
            'date_max': str(sample['date_received'].max()),
            'unique_products': int(sample['product'].nunique()),
            'unique_issues': int(sample['issue'].nunique()),
            'narrative_nonempty_rate': float(sample['consumer_complaint_narrative'].astype(str).str.len().gt(20).mean()),
        })
        if result['required_columns_missing']:
            result['passed'] = False
    except Exception as exc:
        result.update({'passed': False, 'error_type': type(exc).__name__, 'error': str(exc)})

    result['runtime_seconds'] = round(time.perf_counter() - started, 4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get('passed') else 1)


if __name__ == '__main__':
    main()
