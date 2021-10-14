import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.helpers.features import production_rate_dataset, producer_rows_from_df
from src.helpers.figures import plot_helper
from src.models.crmp import CRMP
from src.simulations import injector_names, producer_names


injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])

FIG_DIR = INPUTS['real_data']['figures_dir']
fit_file = INPUTS['real_data']['fit']['sensitivity_analysis']
fit_df = pd.read_csv(fit_file)

predict_file = INPUTS['real_data']['predict']['sensitivity_analysis']
predict_df = pd.read_csv(predict_file)


def plot_production_rate():
    tmp_producer_names = producer_names
    for name in tmp_producer_names:
        producer = producers_df.loc[
            producers_df['Name'] == name
        ]
        production_rate = producer['total rate']
        t = np.linspace(0, len(production_rate), len(production_rate))
        plt.plot(t, production_rate)
        plot_helper(
            FIG_DIR,
            xlabel='Time [days]',
            ylabel='Production Rate [bbls/day]',
            title=name,
            save=True
        )


def plot_injection_rates():
    for name in injector_names:
        injector = injectors_df.loc[
            injectors_df['Name'] == name
        ]
        injection_rate = injector['Water Vol']
        t = np.linspace(0, len(injection_rate), len(injection_rate))
        plt.plot(t, injection_rate)
        plot_helper(
            FIG_DIR,
            xlabel='Time [days]',
            ylabel='Injection Rate [bbls/day]',
            title=name,
            save=True
        )


def plot_production_history_with_fit_and_predict():
    df = pd.read_csv(predict_file)
    starting_index = producer_starting_indicies[1]
    producer = producers[1][starting_index:]
    injectors_tmp = [injector[starting_index:] for injector in injectors]
    X, y = production_rate_dataset(producer, *injectors_tmp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    crmp = CRMP().fit(X_train, y_train)
    for i in range(len(producer_names)):
        producer_df = producer_rows_from_df(df, i+1)
        starting_index = producer_starting_indicies[i]
        producer = producers[i][starting_index:]
        injectors_tmp = [injector[starting_index:] for injector in injectors]
        X, y = production_rate_dataset(producer, *injectors_tmp)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, shuffle=False
        )
        producer_length = len(producer)
        t = np.linspace(1, producer_length, producer_length)
        train_length = len(y_train)
        train_time = t[:train_length]
        test_time = t[train_length:][1:]

        empty = []
        plt.plot(empty, empty, c='r', label='Fit')
        plt.plot(empty, empty, c='g', label='Predict')
        plt.plot(t, producer, c='k')

        for index, row in producer_df.iterrows():
            tau = row['tau_final']
            f1 = row['f1_final']
            f2 = row['f2_final']
            f3 = row['f3_final']
            f4 = row['f4_final']
            crmp.tau_ = tau
            crmp.gains_ = [f1, f2, f3, f4]

            # Fitting
            y_hat = crmp.predict(X_train)
            plt.plot(train_time, y_hat, '--', alpha=0.02, c='r', linewidth=2)

            # Prediction
            y_hat = crmp.predict(X_test)
            plt.plot(test_time, y_hat, ':', alpha=0.02, c='g', linewidth=2)

        plt.vlines(test_time[0], 0, 1.1 * max(producer), linewidth=2, alpha=0.8)
        plot_helper(
            FIG_DIR,
            title=producer_names[i],
            xlabel='Time [days]',
            ylabel='Production Rate [bbls/day]',
            legend=True,
            save=True
        )


def plot_fractional_flow_curve():
    for i in range(len(producer_names)):
        starting_index = producer_starting_indicies[i]
        total_prod = producers[i][starting_index:]
        water_prod = producers_water_production[i][starting_index:]
        t = time[starting_index:]
        water_fraction = water_prod / total_prod
        water_fraction.fillna(0, inplace=True)
        plt.plot(t, water_fraction)
        plot_helper(
            FIG_DIR,
            title=producer_names[i],
            xlabel='Time [days]',
            ylabel='Water Fraction of Total Production [unitless]',
            save=True
        )


# plot_production_rate()
plot_injection_rates()
# plot_production_history_with_fit_and_predict()
# plot_fractional_flow_curve()
