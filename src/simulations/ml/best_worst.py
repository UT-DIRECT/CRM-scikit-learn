import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import (
    BayesianRidge, HuberRegressor, LinearRegression
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from crmp import CrmpBHP, MBBaggingRegressor

from src.config import INPUTS
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import goodness_score
from src.helpers.features import (
    get_real_producer_data, production_rate_dataset, producer_rows_from_df,
    construct_injection_rate_columns, construct_real_production_rate_dataset,
    construct_real_target_vector
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


def best_worse():
    train_sizes = [0.33, 0.735, 0.49, 0.45, 0.52, 0.66, 0.54]
    n_estimators = 100
    delta_t = 1
    models = [
        [CrmpBHP(), False],
        [
            HuberRegressor(
                alpha=0.5,
                epsilon=100,
                fit_intercept=False
            ),
            True
        ],
        [
            LinearRegression(
                fit_intercept=False,
                positive=True
            ),
            False
        ],
    ]
    labels = [
        'CRMP-BHP',
        'Huber Regression (Best)',
        'Linear Regression (Worst)',
    ]

    # for i in [0, 1, 2, 3, 4, 6]:
    for i in [1]:
        # Constructing dataset
        name = producer_names[i]
        print(name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]

        X, y = construct_real_production_rate_dataset(
            producer, injectors, delta_t=delta_t
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_sizes[i], shuffle=False
        )

        train_length = len(X_train)
        t_fit = np.linspace(0, train_length - 1, train_length)
        t_test = np.linspace(train_length, (train_length + 29), 30)

        plt.plot(t_test, y_test[:30], color='k', label='True Value', linewidth=2)

        X_train_scaled = X_train.copy(deep=True)
        X_train_scaled[name] = log_transformation(X_train[name])
        X_test_scaled = X_test.copy(deep=True)
        X_test_scaled[name] = log_transformation(X_test[name])
        y_train_scaled = log_transformation(y_train)
        y_test_scaled = log_transformation(y_test)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        X_train_scaled = X_train_scaled.to_numpy()
        X_test_scaled = X_test_scaled.to_numpy()
        y_train_scaled = y_train_scaled.to_numpy()
        y_test_scaled = y_test_scaled.to_numpy()

        for j in range(len(models)):
            model = models[j][0]
            log = models[j][1]
            print(labels[j])
            bgr = MBBaggingRegressor(
                base_estimator=model, n_estimators=n_estimators, block_size=7,
                bootstrap=True, n_jobs=-1, random_state=1
            )

            if log:
                bgr.fit(X_train_scaled, y_train_scaled)
            else:
                bgr.fit(X_train, y_train)

            if j == 0:
                y_hats = []
                for e in bgr.estimators_:
                    e.q0 = y_train[-1]
                    y_hat_i = e.predict(X_test[:30, 1:])
                    y_hats.append(y_hat_i)
                y_hats_by_time = np.asarray(y_hats).T
                averages = []
                for y_hats_i in y_hats_by_time:
                    average = np.average(y_hats_i)
                    averages.append(average)
                plt.plot(t_test, averages, label=labels[j], alpha=0.5, linewidth=2)
                continue

            y_hats = []
            for e in bgr.estimators_:
                if log:
                    y_hat_i = y_train_scaled[-1]
                else:
                    y_hat_i = y_train[-1]
                y_hat = []
                for k in range(30):
                    if log:
                        X_test_i = X_test_scaled[k, :]
                    else:
                        X_test_i = X_test[k, :]
                    X_test_i[0] = y_hat_i
                    X_test_i = X_test_i.reshape(1, -1)
                    y_hat_i = e.predict(X_test_i)
                    if log:
                        y_hat.append(np.exp(y_hat_i) - 1)
                    else:
                        y_hat.append(y_hat_i)
                y_hats.append(y_hat)
            y_hats_by_time = np.asarray(y_hats).T.reshape(-1, n_estimators)

            averages = []
            p50s = []
            for y_hats_i in y_hats_by_time:
                average = np.average(y_hats_i)
                p50 = np.percentile(y_hats_i, 50)
                averages.append(average)
                p50s.append(p50)

            # Plotting
            p50s = np.array(p50s).clip(min=0)
            averages = np.array(averages).clip(min=0)
            plt.plot(t_test, averages, label=labels[j], alpha=0.5, linewidth=2)

        plt.tight_layout()
        plot_helper(
            FIG_DIR,
            title='{}: 30 Days Prediction for CRMP-BHP and the Best and Worst Performing ML Estimators'.format(name),
            xlabel='Days',
            ylabel='Production Rate [bbls/day]',
            legend=True,
            save=True
        )
        # plt.show()
        print()


def log_transformation(column):
    return np.log(column + 1)

best_worse()
