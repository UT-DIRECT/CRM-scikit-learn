import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers
from src.helpers.figures import plot_helper
from src.visualization import INPUTS


N_sensitivity_analysis_file = INPUTS['crmp']['N_sensitivity_analysis']
FIG_DIR = INPUTS['crmp']['figures_dir']

true_parameters = {
    1: [0.2, 1.5],
    2: [0.4, 1],
    3: [0.6, 5],
    4: [0.8, 50]
}


def plot_parameter_convergence():
    N_sensitivity_analysis_df = pd.read_csv(N_sensitivity_analysis_file)
    for i in range(len(producers)):
        plt.figure(figsize=[7, 4.8])
        producer = i + 1
        producer_rows_df = N_sensitivity_analysis_df.loc[N_sensitivity_analysis_df['Producer'] == producer]
        x_i = producer_rows_df['f1_initial']
        x_f = producer_rows_df['f1_final']
        y_i = producer_rows_df['tau_initial']
        y_f = producer_rows_df['tau_final']
        x = np.array([x_i, x_f]).T
        y = np.array([y_i, y_f]).T
        true_params = true_parameters[producer]
        x_true = true_params[0]
        y_true = true_params[1]
        initial = plt.scatter(x_i, y_i, s=40, c='g', marker='o', label='Initial')
        final = plt.scatter(x_f, y_f, s=40, c='r', marker='x', label='Final')
        for i in range(len(x)):
            plt.plot(x[i], y[i], c='k', alpha=0.3)
        actual = plt.scatter(
            x_true, y_true, s=200, c='b', marker='X',
            label='Actual'
        )
        title = 'ICRMP Producer {}: Initial Parameter Values with Convergence'.format(producer)
        xlabel = 'f1'
        ylabel = 'tau'
        plt.legend(
            handles=[actual, initial, final],
            bbox_to_anchor=(1.04, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


plot_parameter_convergence()
