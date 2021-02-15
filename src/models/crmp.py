import numpy as np

from lmfit import Model, Parameters

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
        self._q2_constructor()
        params = self.fit_production_rate()
        self.tau_ = params[0]
        self.gains_ = params[1:]
        return self


    def ensure_p0(self):
        if self.p0 == []:
            self.p0 = (1. / self.n_gains * np.ones(self.n_gains + 1))
            self.p0[0] = 5


    @property
    def bounds(self):
        lower_bounds = np.zeros(self.n_gains + 1)
        lower_bounds[0] = 1e-6
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


    def fit_production_rate(self):
        model = Model(self.q2, independent_vars=['X'])
        params = Parameters()
        params.add('tau', value=self.p0[0], min=1.e-06, max=100)

        bound = '1'
        for i in range(self.n_gains):
            gain = self.gains[i]
            params.add(gain, value=self.p0[i + 1], min=0, max=1)
            bound += '-{}'.format(gain)
        params.add('bound', value=0, min=0, max=1, expr=bound)

        results = model.fit(self.y_, X=self.X_, params=params, method='powell')
        # Currently 'nelder' method is working the best.
        # Brute force works well too, but is far too slow, takes a minute plus
        # per well. Powell it is.

        pars = []
        pars.append(results.values['tau'])
        for gain in self.gains:
            pars.append(results.values[gain])
        return pars
