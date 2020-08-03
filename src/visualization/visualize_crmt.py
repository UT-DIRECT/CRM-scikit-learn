import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_wfsim import (time, delta_time, qo_tank, w_tank, qw_tank,
    q_tank, f_w, W_t)
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
    pass


def production_rate_vs_injection_rate():
    pass


def production_rate_estimation_and_time_step():
    pass


production_rate_vs_time()
oil_production_rate_vs_time()
water_production_rate_vs_time()
# net_production_vs_time()
# production_rate_vs_injection_rate()
# production_rate_estimation_and_time_step()
