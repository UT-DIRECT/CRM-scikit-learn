from unittest.mock import patch

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted, NotFittedError

from src.helpers.features import production_rate_dataset
from src.models.crm import CRM


# Initializing variables outside of the class since pytest ignores classes
# with a constructor.
q = np.array([2, 3, 4, 5, 6])
inj1 = 2 * q
inj2 = 3 * q
inj3 = 0.5 * q

class TestCRM():


    def test_fit_two_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2)
        crm = CRM().fit(X, y)
        assert(crm.tau_ is not None)
        assert(crm.tau_ > 1 and crm.tau_ < 30)
        assert(crm.gains_ is not None)
        assert(len(crm.gains_) == 2)
        f1 = crm.gains_[0]
        f2 = crm.gains_[1]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        sum_of_gains = f1 + f2
        assert(abs(1 - sum_of_gains) <= 1.e-5)


    def test_fit_three_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        crm = CRM().fit(X, y)
        assert(crm.tau_ is not None)
        assert(crm.tau_ > 1 and crm.tau_ < 30)
        assert(crm.gains_ is not None)
        assert(len(crm.gains_) == 3)
        f1 = crm.gains_[0]
        f2 = crm.gains_[1]
        f3 = crm.gains_[2]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        assert(0 <= f3 <= 1)
        sum_of_gains = f1 + f2 + f3
        assert(abs(1 - sum_of_gains) <= 1.e-5)


    def test_fit_X_y_different_shape(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        X = X.T
        X = X[:-2]
        X = X.T
        with pytest.raises(ValueError):
            crm = CRM().fit(X, y)


    def test_predict_unfitted_crm_raises_error(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        crm = CRM()
        with pytest.raises(NotFittedError):
            crm.predict(X)


    def test_predict_two_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2)
        crm = CRM().fit(X, y)
        # There is no helper to construct the prediction matrix
        # since the prediction matrix is constructed by the cross validator
        X = [q[1:], inj1[1:], inj2[1:]]
        y_hat = crm.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)


    def test_predict_three_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2, inj3)
        crm = CRM().fit(X, y)
        # There is no helper to construct the prediction matrix
        # since the prediction matrix is constructed by the cross validator
        X = [q[1:], inj1[1:], inj2[1:], inj3[1:]]
        y_hat = crm.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)
