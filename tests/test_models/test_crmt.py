import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted, NotFittedError

from src.helpers.features import production_rate_dataset
from src.models.crmt import CRMT


# Initializing variables outside of the class since pytest ignores classes
# with a constructor.
delta_time = np.array([1, 0.5, 2, 2.2, 1.6])
q = np.array([2, 3, 4, 5, 6])
inj = 2 * q

class TestCRMT():


    def test_fit(self):
        X, y = production_rate_dataset(q, delta_time, inj)
        crmt = CRMT().fit(X, y)
        assert(1 < crmt.tau_ < 30)
        assert(0 < crmt.f_r_ < 1)

    def test_predict_before_fit_raises_error(self):
        X, y = production_rate_dataset(q, delta_time, inj)
        crmt = CRMT()
        with pytest.raises(NotFittedError):
            crmt.predict(X)


    def test_predict(self):
        X, y = production_rate_dataset(q, delta_time, inj)
        crmt = CRMT().fit(X, y)
        X = production_rate_dataset(q, delta_time, inj)[0]
        y_hat = crmt.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)
