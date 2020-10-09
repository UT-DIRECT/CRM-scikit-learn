import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


from src.data.read_crmp import actual_parameters, producers
from src.helpers.figures import plot_helper
from src.visualization import INPUTS


q_sensitivity_analysis_file = INPUTS['crmp']['q_sensitivity_analysis']
FIG_DIR = INPUTS['crmp']['figures_dir']


def plot_parameter_convergence():
    q_sensitivity_analysis_df = pd.read_csv(q_sensitivity_analysis_file)
    for i in range(len(producers)):
        fig =  plt.figure()
        ax = Axes3D(fig)
        producer = i + 1
        producer_rows_df = q_sensitivity_analysis_df.loc[q_sensitivity_analysis_df['Producer'] == producer]
        x_i = producer_rows_df['f1_initial']
        x_f = producer_rows_df['f1_final']
        y_i = producer_rows_df['f2_initial']
        y_f = producer_rows_df['f2_final']
        z_i = producer_rows_df['tau_initial']
        z_f = producer_rows_df['tau_final']
        x = np.array([x_i, x_f]).T
        y = np.array([y_i, y_f]).T
        z = np.array([z_i, z_f]).T
        actual_params = actual_parameters[producer]
        x_true = actual_params[0]
        y_true = actual_params[1]
        z_true = actual_params[2]
        initial = ax.scatter(x_i, y_i, z_i, s=40, c='g', marker='o', label='Initial')
        final = ax.scatter(x_f, y_f, z_f, s=40, c='r', marker='x', label='Final')
        for i in range(len(x)):
            ax.plot(x[i], y[i], z[i], c='k', alpha=0.3)
        actual = ax.scatter(
            x_true, y_true, z_true, s=200, c='b', marker='X',
            label='Actual'
        )
        ax.set_title(
            'Producer {}: Initial Parameter Values with Convergence'.format(producer),
            fontsize=14,
        )
        ax.set_xlabel('f1', fontsize=12)
        ax.set_ylabel('f2', fontsize=12)
        ax.set_zlabel('tau', fontsize=12)
        # ax.legend(bbox_to_anchor=(1.04, 1))
        ax.legend(handles=[actual, initial, final], loc="lower left")
        filename = 'producer_{}_initial_parameter_values_with_convergence'.format(producer)
        fig_file = '{}/{}'.format(FIG_DIR, filename)
        plt.savefig(fig_file)


plot_parameter_convergence()
