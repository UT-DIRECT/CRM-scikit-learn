import matplotlib.pyplot as plt
import numpy as np

from lmfit import Model, Parameters
from scipy.optimize import fmin_slsqp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CRM(BaseEstimator, RegressorMixin):


    def __init__(self):
        return


    def fit(self, X=None, y=None):
        X = X.T
        X, y = check_X_y(X, y)
        X = X.T
        self.X_ = X
        self.y_ = y
        n_gains = len(X) - 1
        self.p0 = (1. / n_gains * np.ones(n_gains + 1))
        self.p0[0] = 5
        self._q2_constructor(n_gains)
        lower_bounds = np.zeros(n_gains + 1)
        lower_bounds[0] = 1
        upper_bounds = np.ones(n_gains + 1)
        upper_bounds[0] = 30
        self.bounds = np.array([lower_bounds, upper_bounds]).T.tolist()
        params = self._fit_production_rate(X, y)
        self.tau_ = params[0]
        self.gains_ = params[1:]
        return self


    def predict(self, X):
        check_is_fitted(self)
        return self.q2(X, self.tau_, *self.gains_)


    def _fitting_prior(self, theta):
        # theta[0] = tau
        # theta[1+] = gains, ie: f1, f2, etc
        gains = None
        if len(theta) > 1:
            gains = theta[1:]
        if theta[0] < 1 or theta > 30:
            return 0
        if gains is not None:
            for gain in gains:
                if gain > 1 or gain < 0:
                    return 0
            sum_of_gains = sum(gains)
            if sum_of_gains > 1 or sum_of_gains < 0:
                return 0
        return 1


    def _transition_model(self):
        pass


    def _q2_constructor(self, n_gains):
        gains = ['f{}'.format(i + 1) for i in range(n_gains)]
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
        return sum((self.y_ - self.q2(self.X_, tau, *gains)))


    def _constraints(self, params):
        tau = params[0]
        gains = params[1:]
        return 1 - sum(gains)


    def _fit_production_rate(self, X, y):
        params = fmin_slsqp(
            self._sum_residuals, self.p0, f_eqcons=self._constraints,
            bounds=self.bounds, iter=1000
        )
        return params


    # def _fit_production_rate(self, X, y):
    #     model = Model(self.q2, independent_vars=['X'])
    #     params = Parameters()

    #     params.add('tau', value=5, min=1, max=30)

    #     n_gains = len(X) - 1
    #     print(n_gains)
    #     gains = ['f{}'.format(i + 1) for i in range(n_gains)]
    #     value = 1 / n_gains
    #     expr = '1'
    #     for i in range(n_gains):
    #         name = gains[i]
    #         if i == 0:
    #             params.add(name, value=value, min=0, max=1)
    #         else:
    #             params.add(name, min=0, max=1, expr='1-{}'.format(gains[i-1]))
    #         # if i == n_gains - 1:
    #         #     params.add(name, min=0, expr=expr)
    #         # else:
    #         #     expr += '-{}'.format(name)
    #         #     params.add(name, value=value, min=0, max=1)
    #     # print('expr: ', expr)
    #     # params.add('formation_loss', value=0, vary=False, expr=expr)
    #     print('params: ', params)

    #     results = model.fit(y, X=X, params=params)

    #     pars = []

    #     # print('formation_loss: ', results.values['formation_loss'])
    #     pars.append(results.values['tau'])
    #     for gain in gains:
    #         pars.append(results.values[gain])

    #     return pars
