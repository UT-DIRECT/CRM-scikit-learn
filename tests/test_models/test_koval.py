
import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted, NotFittedError

from src.helpers.features import production_rate_dataset
from src.models.koval import Koval


# Initializing variables outside of the class since pytest ignores classes
# with a constructor.
W_t = np.array([2, 5, 9, 14, 20])
f_w = np.array([0.2, 0.25, 0.35, 0.5, 0.65])

class TestKoval():


    def test_fit(self):
        X = W_t
        y = f_w
        koval = Koval().fit(X, y)
        assert(1000 < koval.V_p_ < np.inf)
        assert(1 < koval.K_val_ < np.inf)


    def test_predict_before_fit_raises_error(self):
        X = W_t
        koval = Koval()
        with pytest.raises(NotFittedError):
            koval.predict(X)


    def test_predict(self):
        X = W_t
        y = f_w
        koval = Koval().fit(X, y)
        y_hat = koval.predict(X)
        assert(y_hat is not None)
        assert(len(y_hat) == 5)
