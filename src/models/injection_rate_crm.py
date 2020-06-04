import copy

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fmin_slsqp
from sklearn.utils.validation import check_is_fitted

from src.models.crm import CRM


class InjectionRateCRM(CRM):


    def predict(self, X):
        check_is_fitted(self)
        X = X.T
        self.X_predict_ = X
        params = self._fit_injection_rate(X)
        self.n_gains = len(self.X_predict_) - 1
        injection_rates = np.reshape(params, (self.n_gains, -1)).tolist()
        X = copy.copy(injection_rates)
        X.insert(0, self.X_predict_[0].tolist())
        X = np.array(X)
        production_rates = self.q2(X, self.tau_, *self.gains_)
        return (production_rates, injection_rates)


    def _objective_function(self, params):
        injection_rates = params
        params = np.reshape(params, (self.n_gains, -1)).tolist()
        X = params
        X.insert(0, self.X_predict_[0].tolist())
        X = np.array(X)
        return -sum(self.q2(X, self.tau_, *self.gains_))


    def _fit_injection_rate(self, X):
        # The CRM function is part of the _sum_residuals function
        time_steps = len(X[0])
        lower_bounds = np.ones(self.n_gains * time_steps) * np.min(X[1:]) * 0.8
        upper_bounds = np.ones(self.n_gains * time_steps) * np.max(X[1:]) * 1.2
        bounds = np.array([lower_bounds, upper_bounds]).T.tolist()
        p0 = np.ones(self.n_gains * time_steps) * np.average(X[1:])
        params = fmin_slsqp(
            self._objective_function, p0, f_eqcons=self._constraints,
            bounds=bounds, iter=10000, iprint=0
        )
        return params
