import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils.validation import check_X_y, check_is_fitted

from src.helpers.features import production_rate_dataset
from src.models.crmp import CRMP


class ICRMP(CRMP):


    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        X = X.T
        self.X_ = X
        self.y_ = y
        q_data = X[1:]
        q_X, q_y = production_rate_dataset(q_data[0], *q_data[1:])
        super().fit(X=q_X, y=q_y)
        return self


    def predict(self, X):
        check_is_fitted(self)
        X = X.T
        N_data = X[0]
        q_data = X[1:]
        q_y = super().predict(q_data.T)
        N2 = self._N2(N_data, q_y)
        return self._N2(N_data, q_y)


    def _N2(self, N_data, q_y):
        return N_data + q_y
