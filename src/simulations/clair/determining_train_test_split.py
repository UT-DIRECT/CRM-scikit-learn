import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from crmp import CRMP, CrmpBHP

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import scorer_for_crmp
from src.helpers.features import (
    get_real_producer_data, construct_real_production_rate_dataset
)
from src.helpers.figures import plot_helper
from src.simulations import injector_names, producer_names


FIG_DIR = INPUTS['real_data']['figures_dir']
injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])


def determine_train_test_split():
    # producer_names = ['PA01', 'PA02', 'PA03', 'PA09', 'PA10', 'PA12']
    train_sizes = np.linspace(0.1, 0.9, 81)
    for i in [4]:
        # Constructing dataset
        name = producer_names[i]
        print(name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(
            producer, injectors
        )
        for train_size in train_sizes:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, shuffle=False
            )
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            train_length = len(X_train)
            t_fit = np.linspace(0, train_length - 1, train_length)
            t_test = np.linspace(train_length, (train_length + 29), 30)

            model = CrmpBHP().fit(X_train, y_train)
            model.q0 = y_train[-1]
            y_hat = model.predict(X_test[:30, 1:])

            plt.plot(t_test, y_test[:30], color='k', label='True Value')
            plt.plot(t_test, y_hat, color='r', label='Prediction')
            plot_helper(
                FIG_DIR,
                title='{}: {} Train Size'.format(name, train_size),
                xlabel='Days',
                ylabel='Production Rate [bbls/day]',
                legend=True,
                save=False
            )
            plt.show()


determine_train_test_split()
