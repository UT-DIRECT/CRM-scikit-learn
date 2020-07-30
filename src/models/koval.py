import numpy as np

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class Koval(BaseEstimator, RegressorMixin):


    def __init__(self):
        return


    def fit(self, X=None, y=None):
        X = X.reshape(-1, 1)
        X, y = check_X_y(X, y)
        V_p, K_val = curve_fit(
            self._koval_field, X, y, bounds=([1000, 1], [1000000, np.inf])
        )[0]
        self.V_p_ = V_p
        self.K_val_ = K_val
        return self


    def predict(self, X):
        check_is_fitted(self)
        return self._koval_field(X, self.V_p_, self.K_val_)


    def _koval_field(self, W_t, V_p, K_val):
        length = len(W_t)
        f_w_hat = np.zeros(length)
        for i in range(length):
            t_D = W_t[i] / V_p
            if t_D < (1 / K_val):
                f = 0
            elif (1 / K_val) < t_D < K_val:
                f = (K_val - np.sqrt(K_val / t_D)) / (K_val - 1)
            else:
                f = 1
            f_w_hat[i] = f
        return f_w_hat
