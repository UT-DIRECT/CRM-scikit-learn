import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.linear_model import (
    BayesianRidge, HuberRegressor, LinearRegression
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from crmp import CRMP, MBBaggingRegressor

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import goodness_score
from src.helpers.features import production_rate_dataset
from src.simulations import injector_names, number_of_producers


stds = [1, 10, 25]
estimators = [CRMP()]
estimators = [
    LinearRegression(), BayesianRidge(), HuberRegressor(), MLPRegressor()
]


def log_transformation(column):
    return np.log(column + 1)


for std in stds:
    print('Standard Deviation: ', std)
    for estimator in estimators:
        print(estimator)
        for i in range(number_of_producers):
            producer = producers[i]
            producer += np.random.normal(loc=0.0, scale=std, size=len(producer))
            X, y = production_rate_dataset(producer, *injectors)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.5, shuffle=False
            )
            model = estimator.fit(X_train, y_train)
            y_hat = []
            # y_hat = model.predict(X_test[:30, 1:])
            y_hat_i = y_train[-1]
            for i in range(30):
                X_test_i = X_test[i, :]
                X_test_i[0] = y_hat_i
                X_test_i = X_test_i.reshape(1, -1)
                y_hat_i = model.predict(X_test_i)
                y_hat.append(y_hat_i)
            r2, mse = fit_statistics(y_hat, y_test[:30])
            print(mse)
            print(min(y_hat))
        print()
    print()
    print()
