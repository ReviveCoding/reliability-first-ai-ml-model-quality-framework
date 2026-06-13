from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description='Validate optional local GPU/NLP dependencies before heavy training.')
    parser.add_argument('--require-cuda', action='store_true', help='Return non-zero when CUDA is unavailable.')
    parser.add_argument('--output', default='outputs/gpu_preflight.json')
    args = parser.parse_args()

    result = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {
            'torch': version('torch'),
            'transformers': version('transformers'),
            'datasets': version('datasets'),
            'sentence_transformers': version('sentence-transformers'),
        },
        'torch_imported': False,
        'cuda_available': False,
        'mps_available': False,
        'device_count': 0,
        'devices': [],
        'tensor_smoke_passed': False,
        'errors': [],
    }
    try:
        import torch
        result['torch_imported'] = True
        result['cuda_available'] = bool(torch.cuda.is_available())
        result['mps_available'] = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        result['device_count'] = int(torch.cuda.device_count()) if result['cuda_available'] else 0
        if result['cuda_available']:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                result['devices'].append({
                    'index': index,
                    'name': torch.cuda.get_device_name(index),
                    'total_memory_gb': round(properties.total_memory / (1024 ** 3), 2),
                    'compute_capability': f'{properties.major}.{properties.minor}',
                })
            device = torch.device('cuda:0')
        elif result['mps_available']:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        x = torch.randn(64, 64, device=device)
        y = x @ x.T
        result['tensor_smoke_passed'] = bool(y.shape == (64, 64) and torch.isfinite(y).all().item())
        result['selected_smoke_device'] = str(device)
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')

    result['gpu_stack_ready'] = bool(
        result['torch_imported'] and result['tensor_smoke_passed'] and
        result['packages']['transformers'] and result['packages']['datasets'] and
        result['packages']['sentence_transformers']
    )
    result['diagnostic_completed'] = True
    result['strict_cuda_ready'] = bool(result['gpu_stack_ready'] and result['cuda_available'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if args.require_cuda and not result['cuda_available']:
        raise SystemExit(1)
    # The GPU stack is optional. The default command is a diagnostic and must
    # not fail a normal CPU/CI audit when Torch or CUDA is intentionally absent.
    raise SystemExit(0)


if __name__ == '__main__':
    main()
