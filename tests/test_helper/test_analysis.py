import numpy as np

from src.helpers.analysis import fit_statistics


def test_fit_statistics_y_hat_too_short():
    y_hat = [1]
    y = [1]
    r2, mse = fit_statistics(y_hat, y)
    assert(r2 in [np.nan])
