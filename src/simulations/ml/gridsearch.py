import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import (
    BayesianRidge, HuberRegressor, LinearRegression
)
from sklearn.model_selection import (
    GridSearchCV, train_test_split, TimeSeriesSplit
)
from sklearn.neural_network import MLPRegressor

from crmp import MBBaggingRegressor

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import scorer
from src.helpers.features import (
    get_real_producer_data, impute_training_data, production_rate_dataset,
    producer_rows_from_df, construct_real_production_rate_dataset
)
from src.helpers.figures import plot_helper
from src.helpers.models import model_namer
from src.simulations import injector_names, producer_names


FIG_DIR = INPUTS['real_data']['figures_dir']
injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])


def train_bagging_regressor_with_different_estimators():
    train_sizes = [0.33, 0.735, 0.49, 0.45, 0.52, 0.66, 0.54]
    for i in [4, 6]:
        # Constructing dataset
        name = producer_names[i]
        print(name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(
            producer, injectors
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_sizes[i], shuffle=False
        )
        X_train[name] = log_transformation(X_train[name])
        X_test[name] = log_transformation(X_test[name])
        y_train_scaled = log_transformation(y_train)
        y_test_scaled = log_transformation(y_test)
        # y_train_scaled = y_train
        # y_test_scaled = y_test
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        y_train_scaled = y_train_scaled.to_numpy()
        y_test_scaled = y_test_scaled.to_numpy()
        n_splits = len(X_train) // 30
        tscv = TimeSeriesSplit(n_splits)
        cv = tscv.split(X_train)

        # Setting up estimator
        bgr = MBBaggingRegressor(
            base_estimator=MLPRegressor(random_state=0), n_estimators=100,
            bootstrap=True, random_state=0
        )
        # LinearRegression hyperparameters
        # param_grid = {
        #     'base_estimator__fit_intercept': [True, False],
        #     'base_estimator__positive': [True, False]
        # }
        # BayesianRidge hyperparamters
        # param_grid = {
        #     'base_estimator__alpha_init': [1e-08],
        #     'base_estimator__lambda_init': [0.1],
        #     'base_estimator__compute_score': [False],
        #     'base_estimator__fit_intercept': [True],
        #     'base_estimator__normalize': [False]
        # }
        # HuberRegressor hyperparamters
        # param_grid = {
        #     'base_estimator__epsilon': [1.35],
        #     'base_estimator__alpha': [0.5],
        #     'base_estimator__fit_intercept': [True]
        # }
        # MLPRegressor hyperparamters
        param_grid = {
            'base_estimator__hidden_layer_sizes': [10],
            'base_estimator__activation': ['logistic'],
            'base_estimator__solver': ['adam'],
        }
        gcv = GridSearchCV(
            bgr, param_grid=param_grid, scoring=scorer, cv=cv
        )

        # Fitting the estimator
        gcv.fit(X_train, y_train_scaled)

        # print(gcv.best_params_)
        best_estimator = gcv.best_estimator_
        best_estimator.fit(X_train, y_train_scaled)
        y_hat_i = y_train_scaled[-1]
        y_hat = []
        for i in range(30):
            X_test_i = X_test[i, :]
            X_test_i[0] = y_hat_i
            X_test_i = X_test_i.reshape(1, -1)
            y_hat_i = best_estimator.predict(X_test_i)
            # y_hat.append(np.exp(y_hat_i) - 1)
            y_hat.append(y_hat_i)

        r2, mse = fit_statistics(y_test[:30], y_hat)
        print(mse)
        # print()
        # print()


# Log transformation prevents negative predictions once predictions are
# converted back to the original scale by taking the exponential.
# https://stackoverflow.com/questions/66334730/ways-to-handle-negative-value-of-prediction-in-regression-model
def log_transformation(column):
    return np.log(column + 1)


train_bagging_regressor_with_different_estimators()
