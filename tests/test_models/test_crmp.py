import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted, NotFittedError

from src.helpers.features import production_rate_dataset
from src.models.crmp import CRMP


# Initializing variables outside of the class since pytest ignores classes
# with a constructor.
q = np.array([2, 3, 4, 5, 6])
inj1 = 2 * q
inj2 = 3 * q
inj3 = 0.5 * q
inj4 = 0.75 * q

class TestCRMP():


    def test_fit_two_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2)
        crmp = CRMP().fit(X, y)
        assert(crmp.tau_ is not None)
        assert(crmp.tau_ > 1 and crmp.tau_ < 100)
        assert(crmp.gains_ is not None)
        assert(len(crmp.gains_) == 2)
        f1 = crmp.gains_[0]
        f2 = crmp.gains_[1]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        sum_of_gains = f1 + f2
        assert(abs(1 - sum_of_gains) <= 1.e-2)


    def test_fit_three_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        crmp = CRMP().fit(X, y)
        assert(crmp.tau_ is not None)
        assert(crmp.tau_ > 1 and crmp.tau_ < 100)
        assert(crmp.gains_ is not None)
        assert(len(crmp.gains_) == 3)
        f1 = crmp.gains_[0]
        f2 = crmp.gains_[1]
        f3 = crmp.gains_[2]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        assert(0 <= f3 <= 1)
        sum_of_gains = f1 + f2 + f3
        assert(abs(1 - sum_of_gains) <= 1.e-2)


    def test_fit_four_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3, inj4)
        crmp = CRMP().fit(X, y)
        assert(crmp.tau_ is not None)
        assert(crmp.tau_ > 1 and crmp.tau_ < 100)
        assert(crmp.gains_ is not None)
        assert(len(crmp.gains_) == 4)
        f1 = crmp.gains_[0]
        f2 = crmp.gains_[1]
        f3 = crmp.gains_[2]
        f4 = crmp.gains_[3]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        assert(0 <= f3 <= 1)
        assert(0 <= f4 <= 1)
        sum_of_gains = f1 + f2 + f3 + f4
        assert(abs(1 - sum_of_gains) <= 1.e-2)


    def test_fit_X_y_different_shape(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        X = X[:-2]
        with pytest.raises(ValueError):
            crmp = CRMP().fit(X, y)


    def test_predict_unfitted_crmp_raises_error(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        crmp = CRMP()
        with pytest.raises(NotFittedError):
            crmp.predict(X)


    def test_predict_two_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2)
        crmp = CRMP().fit(X, y)
        # There is no helper to construct the prediction matrix
        # since the prediction matrix is constructed by the cross validator
        X = np.array([q[1:], inj1[1:], inj2[1:]]).T
        y_hat = crmp.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)


    def test_predict_three_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        crmp = CRMP().fit(X, y)
        # There is no helper to construct the prediction matrix
        # since the prediction matrix is constructed by the cross validator
        X = np.array([q[1:], inj1[1:], inj2[1:], inj3[1:]]).T
        y_hat = crmp.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)
