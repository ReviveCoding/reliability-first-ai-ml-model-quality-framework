import numpy as np

from model_quality.evaluation.calibration_metrics import expected_calibration_error


def test_ece_bounds():
    y = np.array([0,1,1,0])
    p = np.array([[0.8,0.2],[0.2,0.8],[0.3,0.7],[0.6,0.4]])
    ece, table = expected_calibration_error(y, p, n_bins=2)
    assert 0 <= ece <= 1
    assert not table.empty
