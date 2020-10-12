import numpy as np

from scipy.optimize import fmin_slsqp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class CRMP(BaseEstimator, RegressorMixin):


    def __init__(self, p0=0):
        self.p0 = p0


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        self.X_ = X
        self.y_ = y
        self.n_gains = len(X) - 1
        self.ensure_p0()
        self._q2_constructor()
        params = self._fit_production_rate(X, y)
        self.tau_ = params[0]
        self.gains_ = params[1:]
        return self


    def ensure_p0(self):
        if self.p0 == 0:
            self.p0 = (1. / self.n_gains * np.ones(self.n_gains + 1))
            self.p0[0] = 5


    @property
    def bounds(self):
        lower_bounds = np.zeros(self.n_gains + 1)
        lower_bounds[0] = 1
        upper_bounds = np.ones(self.n_gains + 1)
        upper_bounds[0] = 100
        return np.array([lower_bounds, upper_bounds]).T.tolist()


    def predict(self, X):
        X = X.T
        check_is_fitted(self)
        return self.q2(X, self.tau_, *self.gains_)


    def _q2_constructor(self):
        gains = ['f{}'.format(i + 1) for i in range(self.n_gains)]
        _q2 = 'lambda X, tau'

        for gain in gains:
            _q2 += ', {}'.format(gain)

        _q2 += ': X[0] * np.exp(-1 / tau) + (1 - np.exp(-1 / tau)) * ('

        for i in range(len(gains)):
            _q2 += 'X[{}] * {} + '.format((i + 1), gains[i])

        _q2 = _q2[:-2] + ')'

        self.q2 = eval(_q2)


    def _sum_residuals(self, params):
        tau = params[0]
        gains = params[1:]
        return sum((self.y_ - self.q2(self.X_, tau, *gains)) ** 2)


    def _constraints(self, params):
        tau = params[0]
        gains = params[1:]
        return 1 - sum(gains)


    def _fit_production_rate(self, X, y):
        # The CRMP function is part of the _sum_residuals function
        params = fmin_slsqp(
            self._sum_residuals, self.p0, f_eqcons=self._constraints,
            bounds=self.bounds, iter=1000, iprint=0
        )
        return params
