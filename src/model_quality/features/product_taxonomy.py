from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_product_taxonomy(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Product taxonomy config not found: {path}')
    config = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    mapping = config.get('mapping') or {}
    if not isinstance(mapping, dict):
        raise TypeError('product taxonomy mapping must be a dictionary')
    config['mapping'] = {str(k): str(v) for k, v in mapping.items()}
    return config


def canonicalize_product_taxonomy(
    df: pd.DataFrame,
    config_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a versioned CFPB product taxonomy while preserving source labels.

    This is a deterministic domain rule, not a frequency-fitted transformation.
    It can therefore be applied before temporal splitting without looking at
    calibration, selection, or test outcomes.
    """
    config = load_product_taxonomy(config_path)
    target_col = str(config.get('target_column', 'product'))
    raw_col = str(config.get('raw_column', f'{target_col}_raw'))
    mapping = config['mapping']

    if target_col not in df.columns:
        raise KeyError(f'Missing taxonomy target column: {target_col}')

    attrs = dict(df.attrs)
    out = df.copy()
    if raw_col not in out.columns:
        out[raw_col] = out[target_col].fillna('Unknown').astype(str)

    raw = out[raw_col].fillna('Unknown').astype(str)
    canonical = raw.replace(mapping)
    changed = canonical.ne(raw)
    out[target_col] = canonical
    out.attrs.update(attrs)
    out.attrs['product_taxonomy_version'] = config.get('taxonomy_version')

    hit_counts = Counter(raw[changed].astype(str))
    metadata: dict[str, Any] = {
        'taxonomy_version': config.get('taxonomy_version'),
        'source': config.get('source'),
        'description': config.get('description'),
        'config_path': str(Path(config_path)),
        'target_column': target_col,
        'raw_column': raw_col,
        'mapping_count': int(len(mapping)),
        'row_count': int(len(out)),
        'changed_row_count': int(changed.sum()),
        'changed_row_rate': float(changed.mean()) if len(out) else 0.0,
        'raw_class_count': int(raw.nunique(dropna=False)),
        'canonical_class_count': int(canonical.nunique(dropna=False)),
        'mapping_hits': {str(k): int(v) for k, v in sorted(hit_counts.items())},
        'unmapped_raw_labels': sorted(set(raw.astype(str)) - set(mapping)),
    }
    return out, metadata
