from model_quality.models.train_transformer import detect_device


def test_explicit_cpu_is_respected():
    assert detect_device('cpu') == 'cpu'


def test_auto_returns_supported_device_name():
    assert detect_device('auto') in {'cpu', 'cuda', 'mps'}
