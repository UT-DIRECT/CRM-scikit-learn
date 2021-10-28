import warnings
import numpy as np

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from src.models.crmp import CRMP

warnings.filterwarnings('ignore')


class CrmpBHP(CRMP):


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        self.X_ = X
        self.y_ = y
        self.n_gains = len(X) - 2
        self.n = len(X)
        self.gains = ['f{}'.format(i + 1) for i in range(self.n_gains)]
        self.ensure_p0()
        self.bounds = self.ensure_bounds()
        x = self.fit_production_rate()
        self.tau_ = x[0]
        self.J_ = x[1]
        self.gains_ = x[2:]
        return self


    # FIXME: Make sure code works by just using the parent function
    def ensure_p0(self):
        if self.p0 == []:
            self.p0 = (1. / self.n_gains * np.ones(self.n))
            self.p0[0] = 5
            self.p0[1] = 2


    # FIXME: Make sure code works by just using the parent function
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
        return self.production_rate(
            X, self.tau_, self.J_, *self.gains_
        )


    def production_rate(self, X, tau, J, *gains):
        q2 = X[0] * np.exp(-1 / tau)
        pressure_change_term = X[1] * J
        injectors_sum = 0
        for i in range(self.n_gains):
            injectors_sum += X[i + 2] * gains[i]
        q2 += (1 - np.exp(-1 / tau)) * (injectors_sum - pressure_change_term)
        return q2


    def objective(self, x):
        tau = x[0]
        J = x[1]
        gains = x[2:]
        return np.linalg.norm(
            self.y_ - self.production_rate(self.X_, tau, J, *gains)
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
            self.objective, self.p0, hess=self.hess, method='trust-constr',
            bounds=self.bounds,
            constraints=({'type': 'ineq', 'fun': self.constraint}),
            options={'maxiter': 10000}
        ).x
