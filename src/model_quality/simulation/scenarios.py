from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_monte_carlo_config(path: str | Path = 'configs/monte_carlo.yaml') -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Monte Carlo config not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if 'scenarios' not in cfg or not isinstance(cfg['scenarios'], dict):
        raise ValueError('Monte Carlo config must define a scenarios mapping.')
    return cfg


def get_scenario(name: str, path: str | Path = 'configs/monte_carlo.yaml') -> dict:
    cfg = load_monte_carlo_config(path)
    if name not in cfg['scenarios']:
        raise KeyError(f'Unknown scenario {name!r}. Available: {sorted(cfg["scenarios"])}')
    return dict(cfg['scenarios'][name])


def spawn_run_seeds(base_seed: int, n_runs: int) -> list[int]:
    """Create reproducible, independent child seeds for repeated simulations."""
    if n_runs <= 0:
        return []
    seq = np.random.SeedSequence(int(base_seed))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seq.spawn(n_runs)]
