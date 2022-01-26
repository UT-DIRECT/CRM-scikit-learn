import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split

from crmp import CRMP, CrmpBHP, MBBaggingRegressor

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import scorer_for_crmp
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


def train_bagging_regressor_with_crmp():
    # train_sizes = [0.33, 0.735, 0.49, 0.45, 0.52, 0.45, 0.54]
    train_sizes = [0.33, 0.735, 0.49, 0.45, 0.52, 0.66, 0.54]
    # for i in range(len(producer_names) - 1):
    n_estimators = 100
    delta_t = 1
    for i in [0, 1, 2, 3, 4, 6]:
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
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        train_length = len(X_train)
        t_fit = np.linspace(0, train_length - 1, train_length)
        t_test = np.linspace(train_length, (train_length + 29), 30)

        # Setting up estimator
        bgr = MBBaggingRegressor(
            base_estimator=CrmpBHP(delta_t=delta_t), n_estimators=n_estimators,
            block_size=7, bootstrap=True, n_jobs=-1, random_state=0
        )
        bgr.fit(X_train, y_train)
        y_fits = []
        for e in bgr.estimators_:
            y_hat_i = []
            for i in range(len(y_train)):
                e.q0 = X_train[i, 0]
                y_hat_i.append(e.predict(np.array([X_train[i, 1:]])))
            y_fits.append(y_hat_i)
        y_fits_by_time = np.asarray(y_fits).T.reshape(-1, n_estimators)
        y_fits_average = []
        for y_hats_i in y_fits_by_time:
            average = np.average(y_hats_i)
            y_fits_average.append(average)

        r2, mse = fit_statistics(y_fits_average, y_train)

        # Getting all bootstrapped predictions
        y_hats = []
        for e in bgr.estimators_:
            e.q0 = y_train[-1]
            y_hat_i = e.predict(X_test[:30, 1:])
            y_hats.append(y_hat_i)
        y_hats_by_time = np.asarray(y_hats).T
        p10s = []
        averages = []
        p90s = []
        for y_hats_i in y_hats_by_time:
            p10 = np.percentile(y_hats_i, 10)
            average = np.average(y_hats_i)
            p90 = np.percentile(y_hats_i, 90)
            p10s.append(p10)
            averages.append(average)
            p90s.append(p90)

        max_train = np.amax(y_train[-100:])
        max_fit = np.amax(y_fits_average[-100:])
        max_realization = np.amax(y_hats)
        height = max(max_train, max_fit, max_realization)
        # Plotting
        plt.plot(t_fit[-100:], y_train[-100:], color='k')
        plt.plot(t_fit[-100:], y_fits_average[-100:], color='g', label='Fitting')
        plt.plot(t_test, y_test[:30], color='k', label='True Value')
        plt.plot(t_test, averages, color='b', label='Average')
        plt.plot(t_test, p10s, color='r', alpha=0.5, label='P10 & P90')
        plt.plot(t_test, p90s, color='r', alpha=0.5)
        for hat in y_hats:
            plt.plot(t_test, hat, color='k', alpha=0.1)
        plt.annotate(
            'r-squared={:.4f}'.format(r2), xy=(train_length - 60, height)
        )
        plt.vlines(
            train_length - 1, 0, height, linewidth=2, colors='k',
            linestyles='dashed', alpha=0.8
        )
        plot_helper(
            FIG_DIR,
            title='{}: 30 Days Prediction'.format(name),
            xlabel='Days',
            ylabel='Production Rate [bbls/day]',
            legend=True,
            save=True
        )


train_bagging_regressor_with_crmp()
