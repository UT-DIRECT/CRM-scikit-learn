import warnings
import numpy as np

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

warnings.filterwarnings('ignore')


class CRMP(BaseEstimator, RegressorMixin):


    def __init__(self, p0=[]):
        self.p0 = p0


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        self.X_ = X
        self.y_ = y
        self.n_gains = len(X) - 1
        self.n = len(X)
        self.gains = ['f{}'.format(i + 1) for i in range(self.n_gains)]
        self.ensure_p0()
        self.bounds = self.ensure_bounds()
        x = self.fit_production_rate()
        self.tau_ = x[0]
        self.gains_ = x[1:]
        return self


    def ensure_p0(self):
        if self.p0 == []:
            self.p0 = (1. / self.n_gains * np.ones(self.n))
            self.p0[0] = 5
        else:
            self.p0[0] += 10.


    def ensure_bounds(self):
        lower_bounds = np.zeros(self.n)
        upper_bounds = np.ones(self.n)
        lower_bounds[0] = 1e-6
        upper_bounds[0] = 100
        return np.array([lower_bounds, upper_bounds]).T.tolist()


    def predict(self, X):
        X = X.T
        check_is_fitted(self)
        return self.production_rate(X, self.tau_, *self.gains_)


    def production_rate(self, X, tau, *gains):
        q2 = X[0] * np.exp(-1 / tau)
        injectors_sum = 0
        for i in range(self.n_gains):
            injectors_sum += X[i + 1] * gains[i]
        q2 += (1 - np.exp(-1 / tau)) * injectors_sum
        return q2


    def objective(self, x):
        tau = x[0]
        gains = x[1:]
        return np.linalg.norm(
            self.y_ - self.production_rate(self.X_, tau, *gains)
        )


    def constraint(self, x):
        gains = x[1:]
        return 1 - sum(gains)


    @staticmethod
    def hess(x, *args):
        n = len(x)
        return np.zeros((n, n))


    def fit_production_rate(self):
        return minimize(
            self.objective, self.p0, hess=self.hess, method='trust-constr',
            bounds=self.bounds,
            constraints=({'type': 'ineq', 'fun': self.constraint}),
            options={'maxiter': 1000}
        ).x
