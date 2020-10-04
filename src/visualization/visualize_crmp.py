import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import (injectors, net_productions,
    producers, producer_names, Time)
from src.helpers.figures import bar_plot_helper, bar_plot_formater, plot_helper
from src.simulations import step_sizes
from src.visualization import INPUTS


q_fitting_file = INPUTS['crmp']['q_fitting']
q_predictions_metrics_file = INPUTS['crmp']['q_predictions_metrics']
q_predictions_file = INPUTS['crmp']['q_predictions']
N_fitting_file = INPUTS['crmp']['N_fitting']
N_predictions_metrics_file = INPUTS['crmp']['N_predictions_metrics']
N_predictions_file = INPUTS['crmp']['N_predictions']
FIG_DIR = INPUTS['crmp']['figures_dir']

def producers_vs_time():
    plt.figure()
    plt.plot(Time, producers.T)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Production Rate',
        legend=producer_names,
        save=True
    )


def net_production_vs_time():
    plt.figure()
    plt.plot(Time, net_productions.T)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Net Production',
        legend=producer_names,
        save=True
    )


def producers_vs_injector():
    for i in range(len(injectors)):
        plt.figure()
        for producer in producers:
            plt.scatter(injectors[i], producer)
        plot_helper(
            FIG_DIR,
            title='Injector {}'.format(i + 1),
            xlabel='Injection Rate',
            ylabel='Production Rate',
            legend=producer_names,
            save=True
        )


def production_rate_estimators_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(q_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.15
    bar_labels = [
        'CRMP, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse',
        'Lasso, mse', 'Elastic, mse'
    ]
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = predictions_metrics_df.loc[predictions_metrics_df['Producer'] == producer]
        crmp_mse = []
        linear_regression_mse = []
        bayesian_ridge_mse = []
        lasso_mse = []
        elastic_mse = []
        heights = [
            crmp_mse, linear_regression_mse, bayesian_ridge_mse, lasso_mse,
            elastic_mse
        ]
        models = ['CRMP', 'LinearRegression', 'BayesianRidge', 'Lasso', 'ElasticNet']
        for i in range(len(models)):
            mses = producer_rows_df.loc[producer_rows_df['Model'] == models[i]]
            for step_size in step_sizes:
                mse = mses.loc[mses['Step size'] == step_size]['MSE']
                heights[i].append(float(mse))

        title = 'Producer {} '.format(producer)
        xlabel = 'Step Size'
        ylabel = 'Mean Squared Error'
        bar_plot_helper(width, x, x_labels, bar_labels, heights)
        bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel)


def production_rate_good_estimators_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(q_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.23
    bar_labels = [
        'CRMP, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse'
    ]
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = predictions_metrics_df.loc[predictions_metrics_df['Producer'] == producer]
        crmp_mse = []
        linear_regression_mse = []
        bayesian_ridge_mse = []
        heights = [
            crmp_mse, linear_regression_mse, bayesian_ridge_mse
        ]
        models = ['CRMP', 'LinearRegression', 'BayesianRidge']
        for i in range(len(models)):
            mses = producer_rows_df.loc[producer_rows_df['Model'] == models[i]]
            for step_size in step_sizes:
                mse = mses.loc[mses['Step size'] == step_size]['MSE']
                heights[i].append(float(mse))

        title = 'Producer {}'.format(producer)
        xlabel = 'Step Size'
        ylabel = 'Mean Squared Error'
        bar_plot_helper(width, x, x_labels, bar_labels, heights)
        bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel)


def net_production_estimators_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(N_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.15
    bar_labels = [
        'NetCRM, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse',
        'Lasso, mse', 'Elastic, mse'
    ]
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = predictions_metrics_df.loc[predictions_metrics_df['Producer'] == producer]
        crmp_mse = []
        linear_regression_mse = []
        bayesian_ridge_mse = []
        lasso_mse = []
        elastic_mse = []
        heights = [
            crmp_mse, linear_regression_mse, bayesian_ridge_mse, lasso_mse,
            elastic_mse
        ]
        models = ['NetCRM', 'LinearRegression', 'BayesianRidge', 'Lasso', 'ElasticNet']
        for i in range(len(models)):
            mses = producer_rows_df.loc[producer_rows_df['Model'] == models[i]]
            for step_size in step_sizes:
                mse = mses.loc[mses['Step size'] == step_size]['MSE']
                heights[i].append(float(mse))

        title = 'Net Production Estimators Producer {}'.format(producer)
        xlabel = 'Step Size'
        ylabel = 'Mean Squared Error'
        bar_plot_helper(width, x, x_labels, bar_labels, heights)
        bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel)


def net_production_good_estimators_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(N_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.23
    bar_labels = [
        'NetCRM, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse'
    ]
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = predictions_metrics_df.loc[predictions_metrics_df['Producer'] == producer]
        crmp_mse = []
        linear_regression_mse = []
        bayesian_ridge_mse = []
        heights = [
            crmp_mse, linear_regression_mse, bayesian_ridge_mse
        ]
        models = ['NetCRM', 'LinearRegression', 'BayesianRidge']
        for i in range(len(models)):
            mses = producer_rows_df.loc[producer_rows_df['Model'] == models[i]]
            for step_size in step_sizes:
                mse = mses.loc[mses['Step size'] == step_size]['MSE']
                heights[i].append(float(mse))

        title = 'Net Production Estimators Good Estimator MSEs Producer {}'.format(producer)
        xlabel = 'Step Size'
        ylabel = 'Mean Squared Error'
        bar_plot_helper(width, x, x_labels, bar_labels, heights)
        bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel)


def production_rate_with_predictions():
    fitting_df = pd.read_csv(q_fitting_file)
    predictions_df = pd.read_csv(q_predictions_file)
    for i in range(len(producers)):
        producer_number = i + 1
        plt.figure()
        fitting_producer = fitting_df.loc[
            fitting_df['Producer'] == producer_number
        ]
        predictions_producer = predictions_df.loc[
            predictions_df['Producer'] == producer_number
        ]
        producer = producers[i]
        predictions_step_size_2 = predictions_producer.loc[
            predictions_producer['Step size'] == 12
        ]
        models = ['CRMP', 'LinearRegression', 'BayesianRidge']
        # models = ['NetCRM', 'LinearRegression', 'BayesianRidge']
        for model in models:
            fitting = fitting_producer.loc[
                fitting_producer['Model'] == model
            ]
            predictions = predictions_step_size_2.loc[
                predictions_step_size_2['Model'] == model
            ]
            x = [0] * 29
            y = [0] * 29
            for index, row in predictions.iterrows():
                k = int(row['t_i'] - 121)
                x[k] = int(row['t_i'])
                y[k] = row['Prediction']
            x = fitting['t_i'].tolist() + x
            y = fitting['Fit'].tolist() + y
            plt.plot(x, y, linestyle='--', linewidth=2, alpha=0.6)
        plt.axvline(x=120, color='k')
        plt.plot(producer)
        legend = models
        legend.append('Predictions Start')
        legend.append('Data')
        plot_helper(
            FIG_DIR,
            title='Producer {}'.format(producer_number),
            xlabel='Time',
            ylabel='Production Rate Fitting and Predictions',
            legend=legend,
            save=True
        )


def net_production_with_predictions():
    fitting_df = pd.read_csv(N_fitting_file)
    predictions_df = pd.read_csv(N_predictions_file)
    for i in range(len(producers)):
        producer_number = i + 1
        plt.figure()
        fitting_producer = fitting_df.loc[
            fitting_df['Producer'] == producer_number
        ]
        predictions_producer = predictions_df.loc[
            predictions_df['Producer'] == producer_number
        ]
        net_production = net_productions[i]
        predictions_step_size_2 = predictions_producer.loc[
            predictions_producer['Step size'] == 12
        ]
        models = ['NetCRM', 'LinearRegression', 'BayesianRidge']
        # models = ['NetCRM', 'LinearRegression', 'BayesianRidge']
        graphs = []
        for model in models:
            fitting = fitting_producer.loc[
                fitting_producer['Model'] == model
            ]
            predictions = predictions_step_size_2.loc[
                predictions_step_size_2['Model'] == model
            ]
            x = [0] * 29
            y = [0] * 29
            for index, row in predictions.iterrows():
                k = int(row['t_i'] - 121)
                x[k] = int(row['t_i']) + 1
                y[k] = row['Prediction']
            graphs.append(y)
            x = fitting['t_i'].tolist() + x
            y = fitting['Fit'].tolist() + y
            plt.plot(x, y, linestyle='--', linewidth=2, alpha=0.6)
        plt.axvline(x=120, color='k')
        plt.plot(net_production)
        legend = models
        legend.append('Predictions Start')
        legend.append('Data')
        plot_helper(
            FIG_DIR,
            title='Producer {}'.format(producer_number),
            xlabel='Time',
            ylabel='Net Production Predictions',
            legend=legend,
            save=True
        )


def animated_net_production_predictions():
    pass


# producers_vs_time()
# net_production_vs_time()
# producers_vs_injector()
# production_rate_estimators_and_time_steps()
# production_rate_good_estimators_and_time_steps()
# net_production_estimators_and_time_steps()
# net_production_good_estimators_and_time_steps()
# production_rate_with_predictions()
net_production_with_predictions()
