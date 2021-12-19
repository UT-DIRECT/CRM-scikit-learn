from copy import deepcopy

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset, producer_rows_from_df
from src.helpers.figures import plot_helper
from src.helpers.models import model_namer, test_model
from src.models.crmp import CRMP
from src.simulations import number_of_producers, param_grid


def fit_all_producers():
    t = time[1:]
    iterations = 0
    for i in range(number_of_producers):
        producer = producers[i]
        X, y = production_rate_dataset(producer, *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        train_length = len(y_train)
        test_length = len(y_test)
        train_time = t[:train_length]
        test_time = t[train_length:]
        crmp = CRMP(q0=producer[0])
        crmp = crmp.fit(X_train, y_train)
        print('Producer {}'.format(i + 1))
        print('Tau: {}'.format(crmp.tau_))
        print('Gains: {}'.format(crmp.gains_))
        print()


fit_all_producers()
