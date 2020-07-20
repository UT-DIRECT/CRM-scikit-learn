import numpy as np

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class Koval(BaseEstimator, RegressorMixin):


    def __init__(self):
        return


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        V_p, K_val = curve_fit(
            self._koval_field, X, y, bounds=([1000, 1], [inf, inf])
        )
        self.V_p_ = V_p
        self.K_val_ = K_val


    def predict(self, X):
        check_is_fitted(self)
        y = []
        for W_t in X:
            y.append(self._koval_field(W_t, self.V_p_, self.K_val_))
        return y


    def _koval_field(self, W_t, V_p, K_val):
        t_D = W_t / V_p
        if t_D < (1 / K_val):
            return 0
        elif (1 / K_val) < t_D < K_val:
            return (K_val - sqrt(K_val / t_D)) / (K_val - 1)
        else:
            return 1
