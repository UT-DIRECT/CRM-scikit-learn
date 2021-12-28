import warnings
from numba import jit
import numpy as np

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


warnings.filterwarnings('ignore')


@jit(cache=True)
def _production_rate(X, q0, delta_t, tau, J, gains):
    X = X.T
    l = len(X)
    q2 = np.empty(l)
    injection = (1 - np.exp(-delta_t / tau)) * np.sum((X[:, 1:] * gains), axis=1) - (J * tau * X[:, 0]) / delta_t
    for i in range(l):
        q2[i] = q0 * np.exp(-delta_t / tau) + injection[i]
        q0 = q2[i]
    return q2


class CrmpBHP(BaseEstimator, RegressorMixin):


    def __init__(self, q0=0, delta_t=1, p0=[]):
        self.q0 = q0
        self.delta_t = np.int(delta_t)
        self.p0 = p0


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        self.X = X
        self.y = y
        self.n_gains = len(X) - 2
        self.n = len(X)
        self.gains = ['f{}'.format(i + 1) for i in range(self.n_gains)]
        self.ensure_p0()
        self.bounds = self.ensure_bounds()
        x = self.fit_production_rate()
        self.tau_ = x[0]
        self.J_ = x[1]
        self.gains_ = x[2:]
        self.q0 = self.y[-1]
        return self


    def ensure_p0(self):
        if self.p0 == []:
            self.p0 = (1. / self.n_gains * np.ones(self.n))
            self.p0[0] = 5
            self.p0[1] = 2
        elif self.p0[0] < 1e-03:
            self.p0[0] = 1e-03


    def ensure_bounds(self):
        lower_bounds = np.zeros(self.n)
        upper_bounds = np.ones(self.n)
        lower_bounds[0] = 1e-6
        upper_bounds[0] = 100
        lower_bounds[1] = 1e-12
        upper_bounds[1] = 30
        return np.array([lower_bounds, upper_bounds]).T.tolist()


    def predict(self, X):
        X = X.T
        check_is_fitted(self)
        return self._predict_production_rate(
            X, self.tau_, self.J_, *self.gains_
        )


    def _fit_production_rate(self, X, tau, J, *gains):
        X = X.T
        gains = np.array(gains)
        injection = (1 - np.exp(-self.delta_t / tau)) * np.sum((X[:, 2:] * gains), axis=1) - (J * tau * X[:, 1]) / self.delta_t
        q2 = X[:, 0] * np.exp(-self.delta_t / tau) + injection
        return q2


    def _predict_production_rate(self, X, tau, J, *gains):
        gains = np.array(gains)
        return _production_rate(X, self.q0, self.delta_t, tau, J, gains)


    def objective(self, x):
        tau = x[0]
        J = x[1]
        gains = x[2:]
        return np.linalg.norm(
            self.y - self._fit_production_rate(self.X, tau, J, *gains)
        )


    def constraint(self, x):
        gains = x[2:]
        return 1 - sum(gains)


    @staticmethod
    def hess(x, *args):
        n = len(x)
        return np.zeros((n, n))


    def fit_production_rate(self):
        # 10000 iterations is what I need to get reliable convergence, so if
        # I am not getting reliable convergence, first check the number of
        # iterations. I sometimes reduce the number of iterations to make
        # prototyping faster.
        return minimize(
            self.objective, self.p0, method='SLSQP',
            bounds=self.bounds,
            constraints=({'type': 'ineq', 'fun': self.constraint}),
            options={'maxiter': 10000}
        ).x
