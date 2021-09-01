import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_clair import (
    injectors, producers, producer_names, producer_starting_indicies, time
)
from src.helpers.features import production_rate_dataset, producer_rows_from_df
from src.helpers.figures import plot_helper
from src.models.crmp import CRMP

FIG_DIR = INPUTS['real_data']['figures_dir']
fit_file = INPUTS['real_data']['fit']['sensitivity_analysis']
fit_df = pd.read_csv(fit_file)

predict_file = INPUTS['real_data']['predict']['sensitivity_analysis']
predict_df = pd.read_csv(predict_file)


def plot_production_rate():
    tmp_producer_names = ['PA09', 'PA12']
    for name in tmp_producer_names:
        i = producer_names.index(name)
        print(i)
        producer = producers[i]
        starting_index = producer_starting_indicies[i]
        plt.plot(time[starting_index:], producer[starting_index:])
    plot_helper(
        FIG_DIR,
        xlabel='Date',
        ylabel='Production Rate',
        legend=tmp_producer_names,
        save=True
    )


def production_history_with_fit_and_predict():
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
        test_length = len(y_test)
        train_time = t[:train_length]
        test_time = t[train_length:][1:]
        plt.plot(train_time, y_train, c='r', label='Fit')
        plt.plot(test_time, y_test, c='g', label='Predict')
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
            plt.plot(train_time, y_hat, alpha=0.02, c='r', linewidth=2)

            # Prediction
            y_hat = crmp.predict(X_test)
            plt.plot(test_time, y_hat, alpha=0.02, c='g', linewidth=2)

        plt.vlines(test_time[0], 0, 1.1 * max(producer), linewidth=2, alpha=0.8)
        plot_helper(
            FIG_DIR,
            title=producer_names[i],
            xlabel='Time [days]',
            ylabel='Production Rate [bbls/day]',
            legend=True,
            save=True
        )


# plot_production_rate()
production_history_with_fit_and_predict()
