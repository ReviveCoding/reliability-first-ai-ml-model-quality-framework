from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, default=str)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)


def safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict:
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {
            'numpy': package_version('numpy'),
            'pandas': package_version('pandas'),
            'scikit_learn': package_version('scikit-learn'),
            'scipy': package_version('scipy'),
            'lightgbm': package_version('lightgbm'),
            'mlflow': package_version('mlflow'),
            'torch': package_version('torch'),
            'transformers': package_version('transformers'),
            'sentence_transformers': package_version('sentence-transformers'),
        },
    }
