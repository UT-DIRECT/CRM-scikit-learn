import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from crmp import CrmpBHP, MBBaggingRegressor

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import goodness_score
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


def error_and_goodness_vs_amount_of_data():
    percent_of_training_data_used = np.linspace(0.1, 1, 11)
    train_sizes = [0.33, 0.735, 0.49, 0.45, 0.52, 0.66, 0.54]

    # Setting up estimator
    n_estimators = 100
    delta_t = 1
    bgr = MBBaggingRegressor(
        base_estimator=CrmpBHP(delta_t=delta_t), n_estimators=n_estimators,
        block_size=7, bootstrap=True, n_jobs=-1, random_state=0
    )

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

        number_of_days_for_training = len(X_train) * percent_of_training_data_used
        number_of_days_for_training = number_of_days_for_training.astype('int')
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        mses = {'30': [], '90': []}
        goodnesses = {'30': [], '90': []}
        for days in number_of_days_for_training:
            X_train_tmp = X_train[-days:]
            y_train_tmp = y_train[-days:]
            X_train_tmp = X_train_tmp.to_numpy()
            y_train_tmp = y_train_tmp.to_numpy()

            bgr.fit(X_train_tmp, y_train_tmp)

            # Getting all bootstrapped predictions
            y_hats = {'30': [], '90': []}
            for e in bgr.estimators_:
                e.q0 = y_train_tmp[-1]
                y_hat_i = e.predict(X_test[:30, 1:])
                y_hats['30'].append(y_hat_i)
                y_hat_i = e.predict(X_test[:90, 1:])
                y_hats['90'].append(y_hat_i)
            y_hats_30_by_time = np.asarray(y_hats['30']).T
            y_hats_90_by_time = np.asarray(y_hats['90']).T
            averages = {'30': [], '90': []}
            for y_hats_i in y_hats_30_by_time:
                average = np.average(y_hats_i)
                averages['30'].append(average)
            for y_hats_i in y_hats_90_by_time:
                average = np.average(y_hats_i)
                averages['90'].append(average)
            mse_30 = fit_statistics(averages['30'], y_test[:30])[1]
            mse_90 = fit_statistics(averages['90'], y_test[:90])[1]
            goodness_30 = goodness_score(y_test[:30], y_hats_30_by_time.T)
            goodness_90 = goodness_score(y_test[:90], y_hats_90_by_time.T)
            mses['30'].append(mse_30)
            mses['90'].append(mse_90)
            goodnesses['30'].append(goodness_30)
            goodnesses['90'].append(goodness_90)

        plt.plot(number_of_days_for_training, mses['30'], label='30 Days')
        plt.plot(number_of_days_for_training, mses['90'], label='90 Days')
        plot_helper(
            FIG_DIR,
            title='{}'.format(name),
            xlabel='Number of Training Days',
            ylabel='MSE',
            legend=True,
            save=True
        )
        plt.plot(number_of_days_for_training, goodnesses['90'], label='90 Days')
        plot_helper(
            FIG_DIR,
            title='{}'.format(name),
            xlabel='Number of Training Days',
            ylabel='Goodness',
            legend=True,
            save=True
        )


def exponential(x, a, k, b):
    return a * np.exp(x * k) + b


def skewness_vs_goodness():
    df = pd.DataFrame()
    df['skewness'] = [17.3644, 0.9587, 1.3322, 0.7358, 35.2872, 4.9172]
    df['goodness'] = [
        0.5515151515151514, 0.9454545454545454, 0.8848484848484848,
        0.7545454545454545, 0.2787878787878788, 0.4
    ]
    popt = sp.optimize.curve_fit(
        exponential, df['skewness'], df['goodness'], p0=[1, -0.5, 1]
    )[0]
    df['goodness_fit'] = exponential(df['skewness'], *popt)
    df = df.sort_values('skewness')
    r2 = r2_score(df['goodness_fit'], df['goodness'])
    plt.scatter(df['skewness'], df['goodness'], color='r', label='True Values')
    plt.plot(
        df['skewness'], df['goodness_fit'], color='k', linestyle='--',
        label='Fitting'
    )
    plt.text(10, 0.9, '$y = 0.68 e^{-0.40x} + 0.39$')
    plt.text(10, 0.85, 'r-squared = {:.4f}'.format(r2))
    plot_helper(
        FIG_DIR,
        xlabel='Skewness',
        ylabel='Goodness',
        legend=True,
        save=True
    )


# error_and_goodness_vs_amount_of_data()
skewness_vs_goodness()
