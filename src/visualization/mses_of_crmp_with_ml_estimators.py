import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.figures import bar_plot_helper, bar_plot_formater


FIG_DIR = INPUTS['crmp']['figures_dir']

# 30 Days & STD of 25
mses_of_crmp = [
    494.8864224991166, 745.2850276613387, 1407.2848420884102, 8262.242154042546
]
mses_of_linear_regression = [
    547.8848304387975, 947.7425016945487, 2166.7912474156483,
    21108.147016267012
]
mses_of_bayesian_ridge = [
    1185.0186167032145, 1965.846134183957, 3231.5523997618843,
    35161.12423363289
]
mses_of_huber_regressor = [
    1446.4491725082328, 2553.1409860684817, 2850.471453327567, 39578.6089452215
]
mses_of_mlp_regressor = [
    10275.55264333873, 9679.12578851292, 5541.573629303533, 40433.51577318238
]
mses_of_lstm_nn = [
    3955.909695, 8708.158968, 2127.410667, 632.6198996
]


def mses_of_crmp_with_ml_estimators():
    x = np.arange(len(producer_names))
    width = 0.15
    bar_labels = [
        'CRMP', 'Linear Regression', 'Bayesian Ridge Regression',
        'Huber Regression', 'MLP Regression', 'LSTM Neural Network'
    ]
    heights = [
        mses_of_crmp, mses_of_linear_regression, mses_of_bayesian_ridge,
        mses_of_huber_regressor, mses_of_mlp_regressor, mses_of_lstm_nn
    ]
    title = 'Synthetic Data + Noise with STD of 25: Quality of Prediction for 30 Days Using Different Models'
    xlabel = 'Producer'
    ylabel = 'Mean Squared Error'
    bar_plot_helper(width, x, producer_names, bar_labels, heights)
    bar_plot_formater(FIG_DIR, x, producer_names, title, xlabel, ylabel)


mses_of_crmp_with_ml_estimators()
