from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    """Small MLflow wrapper with a filesystem fallback.

    The fallback mirrors the same params/metrics/artifact contract and supports
    both files and directories, so optional Transformer artifacts remain runnable
    even when MLflow is not installed.
    """

    def __init__(self, experiment_name='model_quality_signoff', tracking_uri='mlruns', use_mlflow: bool = True):
        self.experiment_name = experiment_name
        self.tracking_uri = str(tracking_uri)
        self._mlflow = None
        self.initialization_note = ''
        if use_mlflow:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(experiment_name)
            except Exception as exc:
                self._mlflow = None
                self.initialization_note = f'MLflow unavailable; filesystem fallback enabled: {type(exc).__name__}: {exc}'
        if '://' not in self.tracking_uri:
            Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self):
        return self._mlflow is not None

    def log_run(self, run_name: str, params: dict, metrics: dict, artifacts: list[str] | None = None):
        artifacts = artifacts or []
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name) as active_run:
                for key, value in params.items():
                    self._mlflow.log_param(key, value)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        self._mlflow.log_metric(key, float(value))
                for artifact in artifacts:
                    path = Path(artifact)
                    if not path.exists():
                        continue
                    if path.is_dir():
                        self._mlflow.log_artifacts(str(path), artifact_path=path.name)
                    else:
                        self._mlflow.log_artifact(str(path))
                return {'mlflow_enabled': True, 'run_name': run_name, 'run_id': active_run.info.run_id}

        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_id = f'{stamp}_{uuid.uuid4().hex[:8]}'
        out_dir = Path(self.tracking_uri) / 'fallback_runs' / run_name / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'params.json').write_text(json.dumps(params, indent=2, default=str), encoding='utf-8')
        (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str), encoding='utf-8')
        manifest = {
            'run_name': run_name,
            'run_id': run_id,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'mlflow_enabled': False,
            'initialization_note': self.initialization_note,
            'artifacts': [],
        }
        artifact_dir = out_dir / 'artifacts'
        artifact_dir.mkdir(exist_ok=True)
        for artifact in artifacts:
            source = Path(artifact)
            if not source.exists():
                continue
            destination = artifact_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            manifest['artifacts'].append(str(destination.relative_to(out_dir)))
        (out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return {
            'mlflow_enabled': False,
            'run_name': run_name,
            'run_id': run_id,
            'fallback_dir': str(out_dir),
            'note': self.initialization_note,
        }
