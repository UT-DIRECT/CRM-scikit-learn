import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers, true_params
from src.helpers.figures import (
    contour_params, initial_and_final_params_from_df, plot_helper
)
from src.visualization import INPUTS
from src.simulations import number_of_gains, number_of_time_constants


q_fitting_sensitivity_analysis_file = INPUTS['crmp']['crmp']['fit']['sensitivity_analysis']
q_fitting_sensitivity_analysis_df = pd.read_csv(q_fitting_sensitivity_analysis_file)

q_predictions_sensitivity_analysis_file = INPUTS['crmp']['crmp']['predict']['sensitivity_analysis']
q_predictions_sensitivity_analysis_df = pd.read_csv(q_predictions_sensitivity_analysis_file)

best_guesses_fit_file = INPUTS['crmp']['crmp']['fit']['best_guesses']
best_guesses_fit_df = pd.read_csv(best_guesses_fit_file)

best_guesses_predict_file = INPUTS['crmp']['crmp']['predict']['best_guesses']
best_guesses_predict_df = pd.read_csv(best_guesses_predict_file)


FIG_DIR = INPUTS['crmp']['figures_dir']

xlabel ='f1'
ylabel ='tau'


def _producer_rows_from_df(df, producer):
    return df.loc[df['Producer'] == producer]


def parameter_convergence_fitting():
    for i in range(len(producers)):
        plt.figure(figsize=[7, 4.8])
        producer = i + 1
        producer_rows_df = _producer_rows_from_df(
            q_fitting_sensitivity_analysis_df,
            producer
        )
        x, y = initial_and_final_params_from_df(producer_rows_df)
        true_params_tmp = _producer_rows_from_df(true_params, producer)
        x_true = true_params_tmp['f1']
        y_true = true_params_tmp['tau']
        for j in range(len(x)):
            initial = plt.scatter(
                x[j][0], y[j][0], s=40, c='g', marker='o', label='Initial'
            )
            final = plt.scatter(
                x[j][1], y[j][1], s=40, c='r', marker='x', label='Final'
            )
            plt.plot(x[j], y[j], c='k', alpha=0.3)
        actual = plt.scatter(
            x_true, y_true, s=200, c='b', marker='X',
            label='Actual'
        )
        title = 'CRMP Fitting: Producer {} Initial Parameter Values with Convergence'.format(producer)
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


def fitted_params_and_mean_squared_error_fitting():
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = _producer_rows_from_df(
            q_fitting_sensitivity_analysis_df,
            producer
        )
        x, y, z = contour_params(
            producer_rows_df,
            x_column='f1_initial',
            y_column='tau_initial',
            z_column='MSE'
        )
        plt.contourf(x, y, z)
        plt.colorbar()
        true_params_tmp = _producer_rows_from_df(true_params, producer)
        x_true = true_params_tmp['f1']
        y_true = true_params_tmp['tau']
        actual = plt.scatter(x, y, c='red', label='Actual')
        plt.legend(handles=[actual])
        title = 'CRMP Producer {}: Fitted Parameter Values with ln(MSE) from Fitting'.format(producer)
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


def fitted_params_and_mean_squared_error_prediction():
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = _producer_rows_from_df(
            q_predictions_sensitivity_analysis_df,
            producer
        )
        x, y, z = contour_params(
            producer_rows_df,
            x_column='f1_initial',
            y_column='tau_initial',
            z_column='MSE'
        )
        plt.contourf(x, y, z)
        plt.colorbar()
        true_params_tmp = _producer_rows_from_df(true_params, producer)
        x_true = true_params_tmp['f1']
        y_true = true_params_tmp['tau']
        actual = plt.scatter(x, y, c='red', label='Actual')
        plt.legend(handles=[actual])
        title = 'CRMP Producer {}: Fitted Parameter Values with ln(MSE) from Prediction'.format(producer)
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


parameter_convergence_fitting()
fitted_params_and_mean_squared_error_fitting()
fitted_params_and_mean_squared_error_prediction()
