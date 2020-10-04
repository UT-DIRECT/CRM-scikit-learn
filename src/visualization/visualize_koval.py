import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_wfsim import time, f_w, W_t
from src.helpers.figures import bar_plot_helper, bar_plot_formater, plot_helper
from src.simulations import step_sizes
from src.visualization import INPUTS


koval_fitting_file = INPUTS['wfsim']['koval_fitting']
koval_predictions_file = INPUTS['wfsim']['koval_predictions']
koval_predictions_metrics_file = INPUTS['wfsim']['koval_predictions_metrics']
FIG_DIR = INPUTS['wfsim']['figures_dir']


def total_water_injected_and_water_cut():
    plt.figure()
    plt.plot(W_t, f_w)
    plot_helper(
        FIG_DIR,
        xlabel='Total Water Injected',
        ylabel='Water Cut',
        save=True
    )


def water_cut_vs_time():
    plt.figure()
    plt.plot(f_w)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Water Cut',
        save=True
    )


def total_water_injected_and_predicted_water_cut():
    plt.figure()
    fitting_df = pd.read_csv(koval_fitting_file)
    predictions_df = pd.read_csv(koval_predictions_file)
    predictions_step_size_2 = predictions_df.loc[
        predictions_df['Step size'] == 12
    ]
    models = ['Koval', 'LinearRegression', 'ElasticNet']
    # models = ['Koval', 'LinearRegression', 'BayesianRidge', 'Lasso', 'ElasticNet']
    for model in models:
        fitting = fitting_df.loc[
            fitting_df['Model'] == model
        ]
        predictions = predictions_step_size_2.loc[
            predictions_step_size_2['Model'] == model
        ]
        x = [0] * 30
        y = [0] * 30
        for index, row in predictions.iterrows():
            i = int(row['t_i'] - 121)
            x[i] = int(row['t_i'])
            y[i] = row['Prediction']
        x = fitting['t_i'].tolist() + x
        y = fitting['Fit'].tolist() + y
        plt.plot(time[x], y, linestyle='--', linewidth=2, alpha=0.6)
    plt.axvline(x=time[120], color='k')
    plt.plot(time[x], f_w[3:])
    legend = models
    legend.append('Predictions Start')
    legend.append('Data')
    plot_helper(
        FIG_DIR,
        title='Water Cut Fitting and Predictions',
        xlabel='Time',
        ylabel='Estimated Water Cut',
        legend=legend,
        save=True
    )


def total_water_injected_and_predicted_water_cut_dimensionless_time():
    plt.figure()
    V_p = 3.53E+07
    fitting_df = pd.read_csv(koval_fitting_file)
    predictions_df = pd.read_csv(koval_predictions_file)
    predictions_step_size_12 = predictions_df.loc[
        predictions_df['Step size'] == 12
    ]
    models = ['Koval', 'LinearRegression', 'ElasticNet']
    # models = ['Koval', 'LinearRegression', 'BayesianRidge', 'Lasso', 'ElasticNet']
    t_D = [0] * 30
    t_D = W_t / V_p
    for model in models:
        fitting = fitting_df.loc[
            fitting_df['Model'] == model
        ]
        predictions = predictions_step_size_12.loc[
            predictions_step_size_12['Model'] == model
        ]
        x = [0] * 30
        y = [0] * 30
        for index, row in predictions.iterrows():
            i = int(row['t_i'] - 121)
            x[i] = int(row['t_i'])
            y[i] = row['Prediction']
        x = fitting['t_i'].tolist() + x
        y = fitting['Fit'].tolist() + y
        plt.plot(t_D[x], y, linestyle='--', linewidth=2, alpha=0.6)
    plt.axvline(x=t_D[120], color='k')
    plt.plot(t_D[x], f_w[3:])
    legend = models
    legend.append('Predictions Start')
    legend.append('Data')
    plot_helper(
        FIG_DIR,
        title='Water Cut Fitting and Predictions',
        xlabel='Dimensionless Time',
        ylabel='Estimated Water Cut',
        legend=legend,
        save=True
    )


def koval_estimation_error_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(koval_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.15
    bar_labels = [
        'Koval, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse',
        'Lasso, mse', 'Elastic, mse'
    ]
    koval_mse = []
    linear_regression_mse = []
    bayesian_ridge_mse = []
    lasso_mse = []
    elastic_mse = []
    heights = [
        koval_mse, linear_regression_mse, bayesian_ridge_mse, lasso_mse,
        elastic_mse
    ]
    models = ['Koval', 'LinearRegression', 'BayesianRidge', 'Lasso', 'ElasticNet']
    for i in range(len(models)):
        mses = predictions_metrics_df.loc[
            predictions_metrics_df['Model'] == models[i]
        ]
        for step_size in step_sizes:
            height = mses.loc[mses['Step size'] == step_sizes[i]]['MSE']
            heights[i].append(float(height))
    title = 'Koval Fractional Flow Estimators'
    xlabel = 'Step Size'
    ylabel = 'Mean Squared Error'
    bar_plot_helper(width, x, x_labels, bar_labels, heights)
    bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel)


def koval_good_estimators_estimation_error_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(koval_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.15
    bar_labels = [
        'Koval, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse'
    ]
    koval_mse = []
    linear_regression_mse = []
    bayesian_ridge_mse = []
    heights = [
        koval_mse, linear_regression_mse, bayesian_ridge_mse
    ]
    models = ['Koval', 'LinearRegression', 'BayesianRidge']
    for i in range(len(models)):
        mses = predictions_metrics_df.loc[
            predictions_metrics_df['Model'] == models[i]
        ]
        for step_size in step_sizes:
            height = mses.loc[mses['Step size'] == step_sizes[i]]['MSE']
            heights[i].append(float(height))
    title = 'Koval Fractional Flow Good Estimators'
    xlabel = 'Step Size'
    ylabel = 'Mean Squared Error'
    bar_plot_helper(width, x, x_labels, bar_labels, heights)
    bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel)


# total_water_injected_and_water_cut()
# water_cut_vs_time()
total_water_injected_and_predicted_water_cut()
total_water_injected_and_predicted_water_cut_dimensionless_time()
# koval_estimation_error_and_time_steps()
# koval_good_estimators_estimation_error_and_time_steps()
