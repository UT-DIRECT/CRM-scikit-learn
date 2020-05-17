import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted, NotFittedError

from src.helpers.features import net_production_dataset
from src.models.net_crm import NetCRM


# Initializing variables outside of the class since pytest ignores classes
# with a constructor.
N = np.array([2, 5, 9, 14, 20])
q = np.array([2, 3, 4, 5, 6])
inj1 = 2 * q
inj2 = 3 * q
inj3 = 0.5 * q

class TestNetCRM():


    def test_fit_two_injectors(self):
        X, y = net_production_dataset(N, q, inj1, inj2)
        net_crm = NetCRM().fit(X, y)
        assert(net_crm.tau_ is not None)
        assert(net_crm.tau_ > 1 and net_crm.tau_ < 30)
        assert(net_crm.gains_ is not None)
        assert(len(net_crm.gains_) == 2)
        f1 = net_crm.gains_[0]
        f2 = net_crm.gains_[1]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        sum_of_gains = f1 + f2
        assert(abs(1 - sum_of_gains) <= 1.e-5)


    def test_fit_three_injectors(self):
        X, y = net_production_dataset(N, q, inj1, inj2, inj3)
        net_crm = NetCRM().fit(X, y)
        assert(net_crm.tau_ is not None)
        assert(net_crm.tau_ > 1 and net_crm.tau_ < 30)
        assert(net_crm.gains_ is not None)
        assert(len(net_crm.gains_) == 3)
        f1 = net_crm.gains_[0]
        f2 = net_crm.gains_[1]
        f3 = net_crm.gains_[2]
        assert(0 <= f1 <= 1)
        assert(0 <= f2 <= 1)
        assert(0 <= f3 <= 1)
        sum_of_gains = f1 + f2 + f3
        assert(abs(1 - sum_of_gains) <= 1.e-5)


    def test_fit_X_y_different_shape(self):
        X, y = net_production_dataset(N, q, inj1, inj2, inj3)
        X = X[:-2]
        with pytest.raises(ValueError):
            net_crm = NetCRM().fit(X, y)


    def test_predict_unfitted_net_crm_raises_error(self):
        X, y = net_production_dataset(N, q, inj1, inj2, inj3)
        net_crm = NetCRM()
        with pytest.raises(NotFittedError):
            net_crm.predict(X)


    def test_predict_two_injectors(self):
        X, y = net_production_dataset(N, q, inj1, inj2)
        net_crm = NetCRM().fit(X, y)
        # There is no helper to construct the prediction matrix
        # since the prediction matrix is constructed by the cross validator
        X = np.array([N[1:], q[1:], inj1[1:], inj2[1:]]).T
        y_hat = net_crm.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)


    def test_predict_three_injectors(self):
        X, y = net_production_dataset(N, q, inj1, inj2, inj3)
        net_crm = NetCRM().fit(X, y)
        # There is no helper to construct the prediction matrix
        # since the prediction matrix is constructed by the cross validator
        X = np.array([N[1:], q[1:], inj1[1:], inj2[1:], inj3[1:]]).T
        y_hat = net_crm.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 4)
