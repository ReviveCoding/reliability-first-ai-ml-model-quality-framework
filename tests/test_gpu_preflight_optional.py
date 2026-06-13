from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_gpu_preflight_is_optional(tmp_path: Path):
    output = tmp_path / 'gpu.json'
    result = subprocess.run(
        [sys.executable, 'scripts/gpu_preflight.py', '--output', str(output)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0
    assert output.exists()
