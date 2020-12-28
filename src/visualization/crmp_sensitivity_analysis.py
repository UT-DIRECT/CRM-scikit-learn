import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers
from src.helpers.figures import plot_helper
from src.visualization import INPUTS


q_fitting_sensitivity_analysis_file = INPUTS['crmp']['crmp']['fit']['sensitivity_analysis']
q_fitting_sensitivity_analysis_df = pd.read_csv(q_fitting_sensitivity_analysis_file)

q_predictions_sensitivity_analysis_file = INPUTS['crmp']['crmp']['predict']['sensitivity_analysis']
q_predictions_sensitivity_analysis_df = pd.read_csv(q_predictions_sensitivity_analysis_file)

best_guesses_fit_file = INPUTS['crmp']['crmp']['fit']['best_guesses']
best_guesses_fit_df = pd.read_csv(best_guesses_fit_file)

best_guesses_predict_file = INPUTS['crmp']['crmp']['predict']['best_guesses']
best_guesses_predict_df = pd.read_csv(best_guesses_predict_file)

most_predictive_parameters_file = INPUTS['crmp']['crmp']['predict']['most_predictive_parameters']
most_predictive_parameters_df = pd.read_csv(most_predictive_parameters_file)

FIG_DIR = INPUTS['crmp']['figures_dir']

true_parameters = {
    1: [0.2, 1.5],
    2: [0.4, 1],
    3: [0.6, 5],
    4: [0.8, 50]
}

number_of_time_constants = 10
number_of_gains = 6
xlabel ='f1'
ylabel ='tau'


def _producer_rows_from_df(df, producer):
    return df.loc[df['Producer'] == producer]


def _initial_and_final_parameters_from_df(df):
    x_i = df['f1_initial']
    x_f = df['f1_final']
    y_i = df['tau_initial']
    y_f = df['tau_final']
    x = np.array([x_i, x_f]).T
    y = np.array([y_i, y_f]).T
    return (x, y)


def _contour_parameters(df, x_column='', y_column='', z_column=''):
    x = df[x_column].to_numpy()
    x = np.reshape(x, (number_of_time_constants, number_of_gains))
    y = df[y_column].to_numpy()
    y = np.reshape(y, (number_of_time_constants, number_of_gains))
    z = df[z_column].to_numpy()
    z_tmp = []
    for i in z:
        if i == 0:
            z_tmp.append(i)
        else:
            z_tmp.append(np.log(i))
    z = z_tmp
    z = np.reshape(z, (number_of_time_constants, number_of_gains))
    return (x, y, z)


def parameter_convergence_fitting():
    for i in range(len(producers)):
        plt.figure(figsize=[7, 4.8])
        producer = i + 1
        producer_rows_df = _producer_rows_from_df(
            q_fitting_sensitivity_analysis_df,
            producer
        )
        x, y = _initial_and_final_parameters_from_df(producer_rows_df)
        true_params = true_parameters[producer]
        x_true = true_params[0]
        y_true = true_params[1]
        for j in range(len(x)):
            initial = plt.scatter(x[j][0], y[j][0], s=40, c='g', marker='o', label='Initial')
            final = plt.scatter(x[j][1], y[j][1], s=40, c='r', marker='x', label='Final')
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


def fitted_parameters_and_mean_squared_error_fitting():
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = _producer_rows_from_df(
            q_fitting_sensitivity_analysis_df,
            producer
        )
        x, y, z = _contour_parameters(
            producer_rows_df,
            x_column='f_initial',
            y_column='tau_initial',
            z_column='MSE'
        )
        plt.contourf(x, y, z)
        plt.colorbar()
        x, y = true_parameters[producer]
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


def fitted_parameters_and_mean_squared_error_prediction():
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = _producer_rows_from_df(
            q_predictions_sensitivity_analysis_df,
            producer
        )
        x, y, z = _contour_parameters(
            producer_rows_df,
            x_column='f_initial',
            y_column='tau_initial',
            z_column='MSE'
        )
        plt.contourf(x, y, z)
        plt.colorbar()
        x, y = true_parameters[producer]
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


def aggregate_mses_contour_plot():
    x, y, z = _contour_parameters(
        most_predictive_parameters_df,
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


parameter_convergence_fitting()
fitted_parameters_and_mean_squared_error_fitting()
fitted_parameters_and_mean_squared_error_prediction()
aggregate_mses_contour_plot()
