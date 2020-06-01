import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fmin_slsqp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from src.models.crm import CRM


class InjectionRateCRM(CRM):


    def __init__(self):
        return


    def predict(self, X):
        # TODO: This is not done, this needs to optimize the injection rate
        X = X.T
        check_is_fitted(self)
        return self.q2(X, self.tau_, *self.gains_)
