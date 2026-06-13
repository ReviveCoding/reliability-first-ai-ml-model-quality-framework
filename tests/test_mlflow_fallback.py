from pathlib import Path

from model_quality.tracking.mlflow_logger import ExperimentLogger


def test_fallback_logger_copies_file_and_directory_artifacts(tmp_path: Path):
    file_artifact = tmp_path / 'metrics.txt'
    file_artifact.write_text('ok', encoding='utf-8')
    directory_artifact = tmp_path / 'model_dir'
    directory_artifact.mkdir()
    (directory_artifact / 'weights.bin').write_bytes(b'abc')

    logger = ExperimentLogger(tracking_uri=str(tmp_path / 'tracking'), use_mlflow=False)
    result = logger.log_run('transformer', {'epochs': 1}, {'macro_f1': 0.5}, [str(file_artifact), str(directory_artifact)])
    run_dir = Path(result['fallback_dir'])
    assert (run_dir / 'artifacts' / 'metrics.txt').exists()
    assert (run_dir / 'artifacts' / 'model_dir' / 'weights.bin').exists()
    assert (run_dir / 'run_manifest.json').exists()
