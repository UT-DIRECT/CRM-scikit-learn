import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers, true_params
from src.helpers.figures import plot_helper
from src.visualization import INPUTS


sensitivity_analysis_file = INPUTS['crmp']['icrmp']['predict']['sensitivity_analysis']
FIG_DIR = INPUTS['crmp']['figures_dir']


def plot_parameter_convergence():
    sensitivity_analysis_df = pd.read_csv(sensitivity_analysis_file)
    for i in range(len(producers)):
        plt.figure(figsize=[7, 4.8])
        producer = i + 1
        producer_rows_df = sensitivity_analysis_df.loc[sensitivity_analysis_df['Producer'] == producer]
        x_i = producer_rows_df['f1_initial']
        x_f = producer_rows_df['f1_final']
        y_i = producer_rows_df['tau_initial']
        y_f = producer_rows_df['tau_final']
        x = np.array([x_i, x_f]).T
        y = np.array([y_i, y_f]).T
        true_params = true_params[producer]
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


def initial_guesses_and_mean_squared_error():
    sensitivity_analysis_df = pd.read_csv(sensitivity_analysis_file)
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = sensitivity_analysis_df.loc[sensitivity_analysis_df['Producer'] == producer]
        x = producer_rows_df['f1_initial'].to_numpy()
        x = np.reshape(x, (101, 11))
        y = producer_rows_df['tau_initial'].to_numpy()
        y = np.reshape(y, (101, 11))
        z = producer_rows_df['MSE'].to_numpy()
        # z = np.sqrt(z)
        z = np.log(z)
        z = np.reshape(z, (101, 11))
        plt.contourf(x, y, z)
        plt.colorbar()
        x, y = true_params[producer]
        actual = plt.scatter(x, y, c='red', label='Actual')
        plt.legend(handles=[actual])
        title = 'ICRMP: Producer {} Intial Guesses with ln(MSE)'.format(producer)
        xlabel = 'f1'
        ylabel = 'Tau'
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


# plot_parameter_convergence()
initial_guesses_and_mean_squared_error()
