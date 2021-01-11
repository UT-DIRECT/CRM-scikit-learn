import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers, producer_names, Time, true_params
from src.helpers.figures import plot_helper
from src.visualization import INPUTS
from src.simulations import (
    number_of_gains, number_of_producers, number_of_time_constants
)


characteristic_params_file = INPUTS['crmp']['crmp']['predict']['characteristic_params']
characteristic_params_predictions_file = INPUTS['crmp']['crmp']['predict']['characteristic_params_predictions']
characteristic_objective_function_file = INPUTS['crmp']['crmp']['predict']['characteristic_objective_function']

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

# FIXME: This function is found else where
def _contour_params(df, x_column='', y_column='', z_column=''):
    number_of_gains = 20
    number_of_time_constants = 100
    x = df[x_column].to_numpy()
    x = np.reshape(x, (number_of_time_constants, number_of_gains))
    y = df[y_column].to_numpy()
    y = np.reshape(y, (number_of_time_constants, number_of_gains))
    z = df[z_column].to_numpy()
    z_tmp = []
    for i in z:
        # z_tmp.append(i)
        if i == 0:
            z_tmp.append(i)
        else:
            z_tmp.append(np.log10(i))
    z = z_tmp
    z = np.reshape(z, (number_of_time_constants, number_of_gains))
    return (x, y, z)


def characteristic_params_convergence_plot():
    df = pd.read_csv(characteristic_params_file )
    plt.figure(figsize=[7, 4.8])
    characteristic_objective_function_df = pd.read_csv(
        characteristic_objective_function_file
    )
    x, y, z = _contour_params(
        characteristic_objective_function_df, x_column='f1', y_column='tau',
        z_column='MSE'
    )
    x = np.unique(x)
    y = np.unique(y)
    plt.contourf(x, y, z, alpha=1.0, cmap='Oranges')
    plt.colorbar()
    df_producer_rows = df.loc[
        df['Producer'] == 1
    ]
    x, y = _initial_and_final_params_from_df(df_producer_rows)
    for j in range(len(x)):
        initial = plt.scatter(
            x[j][0], y[j][0], s=40, c='g', marker='o', label='Initial', alpha=0.3
        )
        final = plt.scatter(
            x[j][1], y[j][1], s=40, c='r', marker='x', label='Final', alpha=0.5
        )
        plt.plot(x[j], y[j], c='k', alpha=0.15)
    x_true = [true_params[i + 1][0] for i in range(number_of_producers)]
    y_true = [true_params[i + 1][1] for i in range(number_of_producers)]
    actual = plt.scatter(
        x_true, y_true, s=200, c='b', marker='X',
        label='Actual', alpha=1
    )
    title = 'Characteristic Well Parameter Convergence and log(MSE)s from Prediction'
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


def characteristic_well_vs_actual_well():
    characteristic_producer = np.mean(producers, axis=0)
    plt.figure()
    plt.plot(producers.T, alpha=0.3)
    plt.plot(characteristic_producer)
    producer_names.append('Characteristic Well')
    plot_helper(
        FIG_DIR,
        title='Characteristic Well and Actual Producers',
        xlabel='Time',
        ylabel='Production Rate',
        legend=producer_names,
        save=True
    )


def best_characteristic_param():
    df = pd.read_csv(characteristic_params_file )
    df = df.drop('Producer', axis=1)
    df = df.groupby(['tau_initial', 'f1_initial'], axis=0).agg({
        'tau_final': 'first',
        'f1_final': 'first',
        'MSE': 'sum'
    })
    df = df.sort_values('MSE')
    print(df)


# characteristic_params_convergence_plot()
# characteristic_well_vs_actual_well()
best_characteristic_param()
