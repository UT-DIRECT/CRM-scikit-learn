import numpy as np

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class CRMT(BaseEstimator, RegressorMixin):


    def __init__(self):
        return


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        tau, f_r = curve_fit(
            self._crmt, X, y, p0=(5, 0.5), bounds=([1, 0], [30, 1])
        )[0]
        self.tau_ = tau
        self.f_r_ = f_r
        return self


    def predict(self, X):
        check_is_fitted(self)
        return self._crmt(X, self.tau_, self.f_r_)


    def _crmt(self, X, tau, f_r):
        # X[0] = N^k-1
        # X[1] = delta_t
        # X[2] = W_i
        return X[0] * np.exp(-X[1] / tau) + (1 - np.exp(-X[1] / tau)) * (X[2] * f_r)
