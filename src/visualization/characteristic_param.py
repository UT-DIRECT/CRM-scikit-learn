import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers, true_params
from src.helpers.figures import plot_helper
from src.visualization import INPUTS
from src.simulations import (
    number_of_gains, number_of_producers, number_of_time_constants
)


characteristic_params_file = INPUTS['crmp']['crmp']['predict']['characteristic_params']
characteristic_params_predictions_file = INPUTS['crmp']['crmp']['predict']['characteristic_params_predictions']

FIG_DIR = INPUTS['crmp']['figures_dir']

xlabel ='f1'
ylabel ='tau'


# FIXME: This function is found else where
def _initial_and_final_params_from_df(df):
    x_i = df['f1_initial']
    x_f = df['f1_final']
    y_i = df['tau_initial']
    y_f = df['tau_final']
    x = np.array([x_i, x_f]).T
    y = np.array([y_i, y_f]).T
    return (x, y)


def characteristic_params_contour_map():
    df = pd.read_csv(characteristic_params_file)
    df = df.loc[
        df['Producer'] == 1
    ]
    x = df['f1_initial'].to_numpy()
    x = np.reshape(x, (number_of_time_constants, number_of_gains))
    y = df['tau_initial'].to_numpy()
    y = np.reshape(y, (number_of_time_constants, number_of_gains))
    z = df['MSE'].to_numpy()
    z_tmp = []
    for i in z:
        if i == 0:
            z_tmp.append(i)
        else:
            z_tmp.append(np.log(i))
    z = z_tmp
    z = np.reshape(z, (number_of_time_constants, number_of_gains))
    return (x, y, z)


def _get_contour_points(df, x, y):
    x_contour = np.array([])
    y_contour = np.array([])
    z_contour = np.array([])
    for i in range(len(x)):
        f1_initial = x[i][0]
        tau_initial = y[i][0]
        df_at_point = df.loc[
            (df['f1_initial'] == f1_initial) & (df['tau_initial'] == tau_initial)
        ]
        x_contour = np.append(x_contour, f1_initial)
        y_contour = np.append(y_contour, tau_initial)
        z_contour = np.append(z_contour, np.log10(np.sum(df_at_point['MSE'])))
    print(x_contour)
    print(y_contour)
    print(z_contour)
    return (x_contour, y_contour, z_contour)


def characteristic_params_convergence_plot():
    df = pd.read_csv(characteristic_params_file )
    plt.figure(figsize=[7, 4.8])
    df_producer_rows = df.loc[
        df['Producer'] == 1
    ]
    x, y = _initial_and_final_params_from_df(df_producer_rows)
    for j in range(len(x)):
        initial = plt.scatter(x[j][0], y[j][0], s=40, c='g', marker='o', label='Initial')
        final = plt.scatter(x[j][1], y[j][1], s=40, c='r', marker='x', label='Final')
        plt.plot(x[j], y[j], c='k', alpha=0.3)
    x_true = [true_params[i + 1][0] for i in range(number_of_producers)]
    y_true = [true_params[i + 1][1] for i in range(number_of_producers)]
    actual = plt.scatter(
        x_true, y_true, s=200, c='b', marker='X',
        label='Actual', alpha=1
    )
    # x, y, z = _get_contour_points(df, x, y)
    # x = np.unique(x)
    # y = np.unique(y)
    # xv, yv = np.meshgrid(x, y)
    # z = np.reshape(z, (len(y), len(x)))
    # plt.contourf(xv, yv, z, alpha=0.3)
    # plt.colorbar()
    title = 'CRMP Fitting: Characteristic Well Parameter Convergence and log(MSE)s from Prediction'
    plt.legend(
        handles=[actual, initial, final],
        # bbox_to_anchor=(1.04, 1),
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


def aggregate_mses_contour_plot():
    x, y, z = _contour_params(
        characteristic_params_df,
        x_column='f1',
        y_column='tau',
        z_column='aggregate_mses'
    )
    plt.contourf(x, y, z)
    plt.colorbar()
    title = 'CRMP: Most Predictive Parameters in Aggregate'
    plot_helper(
        FIG_DIR,
        title=title,
        xlabel='f1',
        ylabel='tau',
        save=True
    )


characteristic_params_convergence_plot()
