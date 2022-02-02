import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.model_selection import train_test_split

from crmp import CrmpBHP, MBBaggingRegressor

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.features import (
    get_real_producer_data, production_rate_dataset,
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


def calculate_parameter_distribution_bagging():
    block_size = 7
    train_sizes = [0.33, 0.735, 0.49, 0.45, 0.52, 0.66, 0.54]
    n_estimators = 10000
    delta_t = 1
    for i in [0, 1, 2, 3, 4, 6]:
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
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        taus = []
        bgr = MBBaggingRegressor(
            base_estimator=CrmpBHP(delta_t=delta_t), n_estimators=n_estimators,
            block_size=7, bootstrap=True, n_jobs=-1, random_state=0
        )
        bgr.fit(X_train, y_train)

        for e in bgr.estimators_:
            taus.append(e.tau_)

        taus = np.asarray(taus)


        skew = sp.stats.skew(taus)
        print(skew)
        plt.hist(taus, bins=100)
        plt.annotate(
            'Skewness = {:.4f}'.format(skew), xy=(0.6, 0.95),
            xycoords='axes fraction'
        )
        plot_helper(
            FIG_DIR,
            title='{}: Distribution of Taus for Fitting Each Estimator'.format(name),
            xlabel='Taus [days]',
            ylabel='Frequency',
            save=True
        )


def calculate_parameter_distribution_by_blocks():
    block_size = 7
    for i in [0, 1, 2, 3, 4, 6]:
        # Constructing dataset
        name = producer_names[i]
        print(name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(
            producer, injectors
        )

        l = len(X)
        n_blocks = l - block_size + 1
        taus = []
        for i in range(n_blocks):
            X_block = X[i:(i + block_size)]
            y_block = y[i:(i + block_size)]
            model = CrmpBHP().fit(X_block, y_block)
            taus.append(model.tau_)

        skew = sp.stats.skew(taus)
        print(skew)
        continue

        plt.hist(taus, bins=(10000 // 100))
        plot_helper(
            FIG_DIR,
            title='{}: Distribution of Taus for Fitting Each Block of Length 7'.format(name),
            xlabel='Taus [days]',
            ylabel='Frequency',
            save=True
        )




calculate_parameter_distribution_bagging()
# calculate_parameter_distribution_by_blocks()
