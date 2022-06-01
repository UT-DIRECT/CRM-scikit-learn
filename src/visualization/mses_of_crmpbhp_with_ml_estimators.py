import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.figures import bar_plot_helper, bar_plot_formater
from src.simulations import injector_names, producer_names


FIG_DIR = INPUTS['real_data']['figures_dir']
injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])

# 30 Days
mses_of_crmp_bhp = [703816, 1023777, 260772, 406943, 1764213, 4809616]
mses_of_linear_regression = [
    1112746, 122269105, 18808753, 57619439, 11377561, 118126607
]
mses_of_bayesian_ridge = [
    5762649, 8005957, 2763191442012, 1883171, 597884461831, 108110259
]
mses_of_huber_regressor = [
    21499424, 7303603, 17890013, 98375723, 6424738, 117730600
]
mses_of_mlp_regressor = [
    146061964, 72146332, 28438673, 4383430, 12911660, 125483204
]

# 90 Days
# mses_of_crmp_bhp = [3920605, 1212824, 3026316, 1248811, 6041440, 26931403]
# mses_of_linear_regression = [
#     135240307, 78657826, 26619758, 3797895, 14834596, 81461346
# ]
# mses_of_bayesian_ridge = [
#     135261866, 78655745, 26619886, 3810863, 14819005, 81429396
# ]
# mses_of_huber_regressor = [
#     135276553, 78655252, 26617148, 3797063, 14803054, 81386374
# ]
# mses_of_mlp_regressor = [
#     135408617, 78661107, 26607410, 3799160, 14827043, 81497138
# ]


def mses_of_crmpbhp_with_ml_estimators():
    tmp_producer_names = producer_names[:5]
    tmp_producer_names.append(producer_names[6])
    x = np.arange(len(tmp_producer_names))
    width = 0.15
    bar_labels = [
        'CRMP-BHP', 'Linear Regression', 'Bayesian Ridge Regression',
        'Huber Regression', 'MLP Regression'
    ]
    heights = [
        mses_of_crmp_bhp, mses_of_linear_regression, mses_of_bayesian_ridge,
        mses_of_huber_regressor, mses_of_mlp_regressor
    ]
    title = 'North Sea Field: Quality of Prediction for 30 Days Using Different Models'
    xlabel = 'Producer'
    ylabel = 'Mean Squared Error'
    bar_plot_helper(width, x, tmp_producer_names, bar_labels, heights)
    bar_plot_formater(FIG_DIR, x, tmp_producer_names, title, xlabel, ylabel)


mses_of_crmpbhp_with_ml_estimators()
