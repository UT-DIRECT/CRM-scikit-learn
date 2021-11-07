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
from src.helpers.tsboot import mb_bootstrap
from src.models.crmp import CRMP
from src.simulations import number_of_producers, param_grid


FIG_DIR = INPUTS['crmp']['figures_dir']

def mv_bagging():
    n_estimators = 10
    iterations = 0
    attributes = ['tau_', 'gains_']
    for i in [3]: # range(number_of_producers):
        print('Producer {}'.format(i + 1))
        fitted_params = {}
        X, y = production_rate_dataset(producers[i], *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        for n in range(n_estimators):
            X_sampled, y_sampled = mb_bootstrap(X_train, y_train, 15)
            crmp = CRMP().fit(X_sampled, y_sampled)
            for attribute in attributes:
                if attribute not in fitted_params:
                    fitted_params[attribute] = [getattr(crmp, attribute)]
                else:
                    fitted_params[attribute].append(getattr(crmp, attribute))
        tau_ = sum(fitted_params['tau_'])/10
        gains_ = sum(fitted_params['gains_'])/10
        print(tau_)
        print(gains_)
        crmp.tau_ = tau_
        crmp.gains_ = gains_
        t =  time[:-(len(X_train) + 1)]
        plt.plot(t, y_train, c='k', label='True Value')
        plt.plot(t, crmp.predict(X_train), c='g', label='Fit')
        plt.vlines(76, 0, 1000, linewidth=1, alpha=0.8)
        t =  time[-(len(X_test)):]
        plt.plot(t, y_test, c='k')
        plt.plot(t, crmp.predict(X_test), c='r', label='Predict')
        plot_helper(
            FIG_DIR,
            title='Producer {} Bagging'.format(i + 1),
            xlabel='Time [days]',
            ylabel='Production Rate [bbls/days]',
            legend=True,
            save=True
        )
        print()
        print()
        print()

def make_plots():
    tau = [
        1.4995093483335045, 1.0015561627284446, 5.008180134998552,
        49.959803750407175
    ]
    gains = [
        [0.20040581, 0.79901223],
        [0.4009804,  0.59784957],
        [0.60134934, 0.39741433],
        [0.79951319, 0.20004146]
    ]
    for i in range(number_of_producers):
        print('Producer {}'.format(i + 1))
        fitted_params = {}
        X, y = production_rate_dataset(producers[i], *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        crmp = CRMP().fit(X_train, y_train)
        crmp.tau_ = tau[i]
        crmp.gains_ = gains[i]
        t =  time[:-(len(X_train) + 1)]
        plt.plot(t, y_train, c='k', label='True Value')
        plt.plot(t, crmp.predict(X_train), c='g', label='Fit')
        plt.vlines(75, 0, 1000, linewidth=3, alpha=0.8)
        t =  time[-(len(X_test)):]
        plt.plot(t, y_test, c='k')
        plt.plot(t, crmp.predict(X_test), c='r', label='Predict')
        plot_helper(
            FIG_DIR,
            title='Producer {} Bagging'.format(i + 1),
            xlabel='Time [days]',
            ylabel='Production Rate [bbls/days]',
            legend=True,
            save=True
        )
        print()
        print()
        print()


# mv_bagging()
make_plots()
