import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers, producer_names, time, true_params
from src.helpers.figures import (
    contour_params, initial_and_final_params_from_df, plot_helper
)
from src.visualization import INPUTS
from src.simulations import (
    number_of_gains, number_of_producers, number_of_time_constants
)


characteristic_params_file = INPUTS['crmp']['crmp']['predict']['characteristic_params']
characteristic_params_predictions_file = INPUTS['crmp']['crmp']['predict']['characteristic_params_predictions']
characteristic_objective_function_file = INPUTS['crmp']['crmp']['predict']['characteristic_objective_function']
q_predict_sensitivity_analysis_file = INPUTS['crmp']['crmp']['predict']['sensitivity_analysis']

FIG_DIR = INPUTS['crmp']['figures_dir']

xlabel ='f1'
ylabel ='tau'


def characteristic_params_convergence_plot():
    df = pd.read_csv(characteristic_params_file)
    plt.figure(figsize=[7, 4.8])
    characteristic_objective_function_df = pd.read_csv(
        characteristic_objective_function_file
    )
    x, y, z = contour_params(
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
    x, y = initial_and_final_params_from_df(df_producer_rows)
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
    title = 'Characteristic Well Parameter Convergence and ln(MSE)s from Prediction'
    plt.legend(
        handles=[actual, initial, final],
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


def initial_guesses_and_mse_from_prediction():
    df = pd.read_csv(q_predict_sensitivity_analysis_file)
    for i in range(number_of_producers):
        producer = i + 1
        df_producer_rows = df.loc[
            df['Producer'] == producer
        ]
        x, y, z = contour_params(
            df_producer_rows, x_column='f1_initial', y_column='tau_initial',
            z_column='MSE'
        )
        plt.contourf(x, y, z, alpha=1.0)
        plt.colorbar()
        title = 'CRMP: Producer {} Initial Guesses with ln(MSE)s from Prediction'.format(producer)
        x_true = true_params[producer][0]
        y_true = true_params[producer][1]
        actual = plt.scatter(
            x_true, y_true, s=100, c='r',
            label='Actual', alpha=0.5
        )
        plt.legend(
            handles=[actual],
            loc="upper left"
        )
        plt.tight_layout()
        plt.ylim(0, 100)
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


def initial_guesses_and_mse_from_prediction_in_aggregate():
    df = pd.read_csv(q_predict_sensitivity_analysis_file)
    df = df.groupby(['tau_initial', 'f1_initial']).sum().reset_index()
    x, y, z = contour_params(
        df, x_column='f1_initial', y_column='tau_initial', z_column='MSE'
    )
    plt.contourf(x, y, z, alpha=1.0)
    plt.colorbar()
    title = 'CRMP: Initial Guesses with ln(MSE)s from Prediction in Aggregate'
    plt.tight_layout()
    plt.ylim(0, 100)
    plot_helper(
        FIG_DIR,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save=True
    )


characteristic_params_convergence_plot()
characteristic_well_vs_actual_well()
best_characteristic_param()
initial_guesses_and_mse_from_prediction()
initial_guesses_and_mse_from_prediction_in_aggregate()
