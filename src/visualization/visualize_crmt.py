import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_wfsim import (q_tank, Q_t, qo_tank, Qo_t, qw_tank, Qw_t,
    time, w_tank)
from src.helpers.figures import bar_plot_formater, plot_helper
from src.simulations import step_sizes
from src.visualization import INPUTS


crmt_predictions_metrics_file = INPUTS['wfsim']['crmt_predictions_metrics']
FIG_DIR = INPUTS['wfsim']['figures_dir']

def production_rate_vs_time():
    plt.figure()
    plt.plot(time, q_tank)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Production Rate',
        save=True
    )


def oil_production_rate_vs_time():
    plt.figure()
    plt.plot(time, qo_tank)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Oil Production Rate',
        save=True
    )


def water_production_rate_vs_time():
    plt.figure()
    plt.plot(time, qw_tank)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Water Production Rate',
        save=True
    )


def net_production_vs_time():
    plt.figure()
    plt.plot(time, Q_t)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Net Production',
        save=True
    )


def net_oil_production_vs_time():
    plt.figure()
    plt.plot(time, Qo_t)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Net Oil Production',
        save=True
    )


def net_water_production_vs_time():
    plt.figure()
    plt.plot(time, Qw_t)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Net Water Production',
        save=True
    )


def production_rate_vs_injection_rate():
    plt.figure()
    plt.scatter(w_tank, q_tank)
    plot_helper(
        FIG_DIR,
        xlabel='Injection Rate',
        ylabel='Production Rate',
        save=True
    )


def production_rate_estimation_and_time_step():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(crmt_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.25
    heights = []
    for i in range(len(step_sizes)):
        height = predictions_metrics_df.loc[predictions_metrics_df['Step size'] == step_sizes[i]]['MSE']
        heights.append(float(height))
    plt.bar(x, heights, width, label='CRMT, mse', alpha=0.5)
    title = 'CRMT Production Estimators'
    xlabel = 'Step Size'
    ylabel = 'Mean Squared Error'
    bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel, legend=False)


production_rate_vs_time()
oil_production_rate_vs_time()
water_production_rate_vs_time()
net_production_vs_time()
net_oil_production_vs_time()
net_water_production_vs_time()
production_rate_vs_injection_rate()
production_rate_estimation_and_time_step()
