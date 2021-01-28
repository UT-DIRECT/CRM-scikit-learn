import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helpers.figures import plot_helper
from src.visualization import INPUTS


FIG_DIR = INPUTS['crmp']['figures_dir']
tau_at_zero_file = INPUTS['crmp']['crmp']['tau_at_zero']
tau_at_zero_df = pd.read_csv(tau_at_zero_file)


def production_rate_vs_different_time_constants():
    time = tau_at_zero_df['time']
    taus = tau_at_zero_df.iloc[:, 2:]
    plt.plot(time, taus, alpha=0.5, linewidth=3)
    plot_helper(
        FIG_DIR,
        title='CRMP: Constant Injection Rate Over Different Time Constants',
        xlabel='Time',
        ylabel='Production Rate',
        legend=[
            'Tau = 1e-06', 'Tau = 1', 'Tau = 10', 'Tau = 20', 'Tau = 50',
            'Tau = 100'
        ],
        save=True
    )

production_rate_vs_different_time_constants()
