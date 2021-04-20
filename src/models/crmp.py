import numpy as np
# import autograd.numpy as np

from autograd import grad
from lmfit import Model, Parameters
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class CRMP(BaseEstimator, RegressorMixin):


    def __init__(self, p0=[]):
        self.p0 = p0


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        self.X_ = X
        self.y_ = y
        self.n_gains = len(X) - 1
        self.gains = ['f{}'.format(i + 1) for i in range(self.n_gains)]
        self.ensure_p0()
        self.bounds = self.ensure_bounds()
        params = self.fit_production_rate()
        self.tau_ = params[0]
        self.gains_ = params[1:]
        return self


    def ensure_p0(self):
        if self.p0 == []:
            self.p0 = (1. / self.n_gains * np.ones(self.n_gains + 1))
            self.p0[0] = 5
        else:
            self.p0[0] += 10.


    def ensure_bounds(self):
        lower_bounds = np.zeros(self.n_gains + 1)
        upper_bounds = np.ones(self.n_gains + 1)
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


    def objective(self, params):
        tau = params[0]
        gains = params[1:]
        return np.linalg.norm(
            self.y_ - self.production_rate(self.X_, tau, *gains)
        )


    def constraint(self, params):
        gains = params[1:]
        return 1 - sum(gains)


    def fit_production_rate(self):
        # , jac=grad(self.objective)
        return minimize(
            self.objective, self.p0, method='SLSQP',
            bounds=self.bounds,
            constraints=({'type': 'ineq', 'fun': self.constraint}),
        ).x
