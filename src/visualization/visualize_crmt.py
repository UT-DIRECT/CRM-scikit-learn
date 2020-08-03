import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_wfsim import (delta_time, f_w, q_tank, Q_t, qo_tank, Qo_t,
     qw_tank, Qw_t, time, w_tank, W_t)
from src.helpers.figures import bar_plot_helper, bar_plot_helper, plot_helper
from src.visualization import INPUTS

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
    pass


production_rate_vs_time()
oil_production_rate_vs_time()
water_production_rate_vs_time()
net_production_vs_time()
net_oil_production_vs_time()
net_water_production_vs_time()
production_rate_vs_injection_rate()
# production_rate_estimation_and_time_step()
