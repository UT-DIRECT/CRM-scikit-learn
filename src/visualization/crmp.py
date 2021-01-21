import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import (injectors, net_productions,
    producers, producer_names, time)
from src.helpers.figures import bar_plot_helper, bar_plot_formater, plot_helper
from src.simulations import step_sizes
from src.visualization import INPUTS


fit_file = INPUTS['crmp']['crmp']['fit']['fit']
metrics_file = INPUTS['crmp']['crmp']['predict']['metrics']
predict_file = INPUTS['crmp']['crmp']['predict']['predict']
FIG_DIR = INPUTS['crmp']['figures_dir']

def producers_vs_time():
    plt.figure()
    plt.plot(time, producers.T)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Production Rate',
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
    metrics_df = pd.read_csv(metrics_file)
    x = np.arange(len(x_labels))
    width = 0.15
    bar_labels = [
        'CRMP, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse',
        'Lasso, mse', 'Elastic, mse'
    ]
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = metrics_df.loc[metrics_df['Producer'] == producer]
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
    metrics_df = pd.read_csv(metrics_file)
    x = np.arange(len(x_labels))
    width = 0.23
    bar_labels = [
        'CRMP, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse'
    ]
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = metrics_df.loc[metrics_df['Producer'] == producer]
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


def production_rate_with_predictions():
    fit_df = pd.read_csv(fit_file)
    predict_df = pd.read_csv(predict_file)
    for i in range(len(producers)):
        producer_number = i + 1
        plt.figure()
        fitting_producer = fit_df.loc[
            fit_df['Producer'] == producer_number
        ]
        predictions_producer = predict_df.loc[
            predict_df['Producer'] == producer_number
        ]
        producer = producers[i]
        predictions_step_size_2 = predictions_producer.loc[
            predictions_producer['Step size'] == 12
        ]
        models = ['CRMP', 'LinearRegression', 'BayesianRidge']
        # models = ['ICRMP', 'LinearRegression', 'BayesianRidge']
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


# producers_vs_time()
# producers_vs_injector()
# production_rate_estimators_and_time_steps()
# production_rate_good_estimators_and_time_steps()
# production_rate_with_predictions()
# tau_at_zero()
