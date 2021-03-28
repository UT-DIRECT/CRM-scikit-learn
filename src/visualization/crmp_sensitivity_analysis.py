import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers, true_params
from src.helpers.features import producer_rows_from_df
from src.helpers.figures import (
    contour_params, initial_and_final_params_from_df, plot_helper
)
from src.visualization import INPUTS
from src.simulations import (
    number_of_gains, number_of_producers, number_of_time_constants
)


q_fitting_sensitivity_analysis_file = INPUTS['crmp']['crmp']['fit']['sensitivity_analysis']
q_fitting_sensitivity_analysis_df = pd.read_csv(q_fitting_sensitivity_analysis_file)

q_predictions_sensitivity_analysis_file = INPUTS['crmp']['crmp']['predict']['sensitivity_analysis']
q_predictions_sensitivity_analysis_df = pd.read_csv(q_predictions_sensitivity_analysis_file)

best_guesses_fit_file = INPUTS['crmp']['crmp']['fit']['best_guesses']
best_guesses_fit_df = pd.read_csv(best_guesses_fit_file)

best_guesses_predict_file = INPUTS['crmp']['crmp']['predict']['best_guesses']
best_guesses_predict_df = pd.read_csv(best_guesses_predict_file)

objective_function_file = INPUTS['crmp']['crmp']['predict']['objective_function']
objective_function_df = pd.read_csv(objective_function_file )


FIG_DIR = INPUTS['crmp']['figures_dir']

xlabel ='f1'
ylabel ='tau'


def parameter_convergence():
    for i in range(len(producers)):
        plt.figure(figsize=[7, 4.8])
        producer = i + 1
        producer_rows_df = producer_rows_from_df(
            q_fitting_sensitivity_analysis_df,
            producer
        )
        x, y = initial_and_final_params_from_df(producer_rows_df)
        x_true, y_true = true_params[producer]
        for j in range(len(x)):
            initial = plt.scatter(
                x[j][0], y[j][0], s=40, c='g', marker='o', label='Initial'
            )
            final = plt.scatter(
                x[j][1], y[j][1], s=40, c='r', marker='x', label='Final'
            )
            plt.plot(x[j], y[j], c='k', alpha=0.15)
        actual = plt.scatter(
            x_true, y_true, s=200, c='b', marker='X',
            label='True Value'
        )
        # actual = plt.scatter(
        #     x_true, y_true, s=100, c='r', label='Actual', alpha=0.5
        # )
        title = 'CRMP: Producer {} Initial Parameter Values with Convergence'.format(producer)
        plt.legend(
            handles=[actual, initial, final],
            bbox_to_anchor=(1.04, 1),
            loc="upper left"
        )
        plt.xlim(0, 1)
        plt.ylim(0, 100)
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
        producer_rows_df = producer_rows_from_df(
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
        x, y = true_params[producer]
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
        producer_rows_df = producer_rows_from_df(
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
        x, y = true_params[producer]
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


def initial_guesses_and_mse_from_prediction():
    df = q_fitting_sensitivity_analysis_df
    for i in range(number_of_producers):
        producer = i + 1
        df_producer_rows = df.loc[
            df['Producer'] == producer
        ]
        x, y, z = contour_params(
            df_producer_rows, x_column='f1_initial', y_column='tau_initial',
            z_column='MSE'
        )
        plt.contourf(x, y, z, 15, alpha=1.0)
        plt.colorbar()
        title = 'CRMP: Producer {} Initial Guesses with MSEs from Prediction'.format(producer)
        x_true, y_true = true_params[producer]
        actual = plt.scatter(
            x_true, y_true, s=100, c='r', label='Actual', alpha=0.5
        )
        plt.legend(handles=[actual], loc='upper left')
        plt.tight_layout()
        plt.ylim(0, 100)
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


def objective_function_contour_plot():
    for i in range(number_of_producers):
        producer = i + 1
        producer_df = producer_rows_from_df(
            objective_function_df, producer
        )
        x, y, z = contour_params(
            producer_df , x_column='f1', y_column='tau',
            z_column='MSE'
        )
        plt.contourf(x, y, z, 15, alpha=1.0)
        plt.colorbar()
        title = 'CRMP: Producer {} Objective Function'.format(producer)
        x_true, y_true = true_params[producer]
        actual = plt.scatter(
            x_true, y_true, s=100, c='r', label='True Value', alpha=0.4
        )
        plt.legend(handles=[actual], loc='upper left')
        plt.tight_layout()
        plt.ylim(0, 100)
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


def gradient_across_parameter_space_prediction_data():
    for i in range(number_of_producers):
        producer = i + 1
        producer_df = producer_rows_from_df(
            objective_function_df, producer
        )
        x, y, z = contour_params(
            producer_df , x_column='f1', y_column='tau',
            z_column='MSE'
        )
        dz = np.gradient(z)[0]
        plt.contourf(x, y, dz, alpha=1.0)
        plt.colorbar()
        title = 'CRMP: Producer {} Gradient Across Parameter Space for MSEs from Prediction'.format(producer)
        plt.tight_layout()
        plt.ylim(0, 100)
        plot_helper(
            FIG_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save=True
        )


# parameter_convergence()
# fitted_params_and_mean_squared_error_fitting()
# fitted_params_and_mean_squared_error_prediction()
# initial_guesses_and_mse_from_prediction()
objective_function_contour_plot()
# gradient_across_parameter_space_prediction_data()
