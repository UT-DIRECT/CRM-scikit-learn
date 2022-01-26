import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def calculate_parameter_distribution_by_blocks():
    # producer_names = ['PA01', 'PA02', 'PA03', 'PA09', 'PA10', 'PA12']
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

        plt.hist(taus, bins=(10000 // 100))
        plot_helper(
            FIG_DIR,
            title='{}: Distribution of Taus for Fitting Each Block of Length 7'.format(name),
            xlabel='Taus [days]',
            ylabel='Frequency',
            save=True
        )



calculate_parameter_distribution_by_blocks()
