from copy import deepcopy

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crmp import CRMP
from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.helpers.features import (
    construct_change_in_pressure_column, get_real_producer_data,
    production_rate_dataset, producer_rows_from_df
)
from src.helpers.features import impute_training_data
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
        l = len(injection_rate)
        t = np.linspace(0, l, l)
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


def plot_average_hour_production_rate():
    t = np.linspace(1, 1317, 1317)
    for name in producer_names:
        producer = producers_df.loc[producers_df['Name'] == name]
        production_rate = producer['Total Vol']
        on_line_hours = producer['On-Line']
        hourly_production_rate = production_rate / on_line_hours
        hourly_production_rate.fillna(0, inplace=True)
        hourly_production_rate.replace(np.inf, 0, inplace=True)
        l = len(hourly_production_rate)
        plt.plot(t[-l:], hourly_production_rate)
        y_max = 1.1 * max(hourly_production_rate)
        print(y_max)
        plt.ylim(0, y_max)
        plot_helper(
            FIG_DIR,
            title=name,
            xlabel='Time [days]',
            ylabel='Hourly Production Rate [bbls/hour]',
            save=True
        )


def plot_on_line_hours_per_day():
    for name in producer_names:
        producer = producers_df.loc[producers_df['Name'] == name]
        production_rate = producer['Total Vol']
        on_line_hours = producer['On-Line']
        plt.hist(on_line_hours, bins=5)
        plot_helper(
            FIG_DIR,
            title=name,
            xlabel='Time [days]',
            save=True
        )


def plot_histogram_of_production_rates():
    for name in producer_names:
        producer = producers_df.loc[producers_df['Name'] == name]
        production_rate = producer[producer['Total Vol'] != 0]['Total Vol']
        plt.hist(production_rate, bins=10)
        plot_helper(
            FIG_DIR,
            title=name,
            xlabel='Production Rate [bbls/day]',
            save=True
        )


def plot_bhp():
    for name in producer_names:
        producer = producers_df.loc[producers_df['Name'] == name]
        bhp = producer['Av BHP']
        l = len(bhp)
        t = np.linspace(1, l, l)
        plt.plot(t, bhp)
        plot_helper(
            FIG_DIR,
            title=name,
            xlabel='Time [days]',
            ylabel='Bottom Hole Pressure [psi]',
            save=True
        )


def plot_delta_bhp():
    for name in producer_names:
        producer = get_real_producer_data(producers_df, name, bhp=True)
        delta_p = producer['delta_p']
        l = len(delta_p)
        t = np.linspace(1, l, l)
        plt.plot(t, delta_p)
        plot_helper(
            FIG_DIR,
            title=name,
            xlabel='Time [days]',
            ylabel='Change in Bottom Hole Pressure [psi]',
            save=True
        )


def plot_imputed_and_original_production_rate():
    for name in producer_names:
        producer = get_real_producer_data(producers_df, name)
        original_data = deepcopy(producer[name])
        l = len(producer)
        y = np.zeros(l)
        impute_training_data(producer, y, name)[0]
        t = np.linspace(1, l, l)
        plt.plot(t, original_data)
        plt.plot(t, producer[name])
        plot_helper(
            FIG_DIR,
            title='{}: Imputed Production Data'.format(name),
            xlabel='Time [days]',
            ylabel='Producer Rate [bbls/day]',
            save=True
        )


def plot_production_rate_and_injection_rate():
    for producer_name in ['PA12']:
        producer = get_real_producer_data(producers_df, producer_name)
        injectors = []
        for name in injector_names:
            injector = injectors_df.loc[
                injectors_df['Name'] == name,
                ['Water Vol', 'Date']
            ]
            injectors.append(injector)
        plt.plot(producer['Date'], producer[producer_name], alpha=0.5)
        for i in range(len(injectors)):
            injector = injectors[i]
            plt.plot(
                injector['Date'], injector['Water Vol'], alpha=0.5,
                label='Injector {}'.format(i + 1)
            )
        dates = producer['Date'].tolist()
        middle_date = dates[int(len(dates) * 0.40)]
        plt.vlines(middle_date, 0, 50000, linewidth=1, alpha=0.8)
        print(middle_date)
        plt.gcf().autofmt_xdate()
        plt.title(producer_name)
        plt.xlabel('Dates')
        plt.ylabel('Production Rate [bbls/day]')
        plt.legend()
        plt.show()



# plot_production_rate()
# plot_injection_rates()
# plot_production_history_with_fit_and_predict()
# plot_fractional_flow_curve()
# plot_average_hour_production_rate()
# plot_on_line_hours_per_day()
# plot_histogram_of_production_rates()
# plot_bhp()
# plot_delta_bhp()
# plot_imputed_and_original_production_rate()
plot_production_rate_and_injection_rate()
