from unittest.mock import patch

import numpy as np
from sklearn.utils.validation import check_is_fitted

from src.helpers.features import production_rate_dataset
from src.models.crm import CRM


q = np.array([2, 3, 4, 5, 6])
inj1 = 2 * q
inj2 = 3 * q
inj3 = 0.5 * q

class TestCRM():


    def test_fit_two_injectors(self):
        X, y = production_rate_dataset(q, inj1, inj2)
        crm = CRM().fit(X, y)
        assert(crm.tau_ is not None)
        print(crm.tau_)
        assert(crm.tau_ > 1 and crm.tau_ < 30)
        assert(crm.gains_ is not None)
        assert(len(crm.gains_) == 2)
        f1 = crm.gains_[0]
        f2 = crm.gains_[1]
        assert(f1 >= 0 and f1 <= 1)
        assert(f2 >= 0 and f2 <= 1)
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
        assert(f1 >= 0 and f1 <= 1)
        assert(f2 >= 0 and f2 <= 1)
        assert(f3 >= 0 and f3 <= 1)
        sum_of_gains = f1 + f2 + f3
        assert(abs(1 - sum_of_gains) <= 1.e-5)
