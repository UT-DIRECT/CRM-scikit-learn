import matplotlib.pyplot as plt
import numpy as np

from lmfit import Model, Parameters
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CRM(BaseEstimator, RegressorMixin):


    def __init__(self):
        return


    def fit(self, X=None, y=None):
        X = X.T
        X, y = check_X_y(X, y)
        X = X.T
        n_gains = len(X) - 1
        self._q2_constructor(n_gains)
        params = self.fit_production_rate(X, y)
        self.tau_ = params[0]
        self.gains_ = params[1:]
        return self


    def predict(self, X):
        check_is_fitted(self)
        return self.q2(X, self.tau_, *self.gains_)


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


    def fit_production_rate(self, X, y):
        model = Model(self.q2, independent_vars=['X'])
        params = Parameters()

        params.add('tau', value=5, min=1, max=30)

        n_gains = len(X) - 1
        gains = ['f{}'.format(i + 1) for i in range(n_gains)]
        value = 1 / n_gains
        expr = '1'
        for i in range(n_gains):
            name = gains[i]
            params.add(name, value=value, min=0, max=1)
            expr += '-{}'.format(name)
        params.add('formation_loss', value=0, vary=False, expr=expr)

        results = model.fit(y, X=X, params=params)

        pars = []
        f1 = results.values['f1']
        f2 = results.values['f2']
        gain = f1 + f2

        pars.append(results.values['tau'])
        for gain in gains:
            pars.append(results.values[gain])

        return pars
