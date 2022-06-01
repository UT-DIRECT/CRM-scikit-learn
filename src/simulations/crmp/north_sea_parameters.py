import numpy as np
import pandas as pd

from crmp import CRMP

from src.helpers.features import (
    get_real_producer_data, construct_real_production_rate_dataset
)
from src.simulations import producer_names


def average_parameters_for_clair():
    for name in producer_names:
        print(name)
        producer = get_real_producer_data(producers_df, name)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(producer, injectors, bhp=None)
        # X = X.drop(columns='IA13')
        model = CRMP().fit(X, y)
        print('Tau: ', model.tau_)
        print('Gains: ', model.gains_)
        print()
        print()

average_parameters_for_clair()
