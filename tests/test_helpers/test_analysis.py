import numpy as np

from src.helpers.analysis import fit_statistics


class TestFitStatistics():


    def test_not_enough_data(self):
        y_hat = [1]
        y = [1]
        r2, mse = fit_statistics(y_hat, y)
        assert(r2 in [np.nan])


    def test_fit_statistics(self):
        y_hat = [1, 2, 3]
        y = [1, 3, 4]
        stats = fit_statistics(y_hat, y)
        assert(len(stats) == 2)
        assert(None not in stats)
