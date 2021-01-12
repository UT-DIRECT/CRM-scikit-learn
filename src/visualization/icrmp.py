import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import (net_productions, producer_names, time)
from src.helpers.figures import bar_plot_helper, bar_plot_formater, plot_helper
from src.simulations import step_sizes
from src.visualization import INPUTS


fit_file = INPUTS['crmp']['icrmp']['fit']['fit']
metrics_file = INPUTS['crmp']['icrmp']['predict']['metrics']
predict_file = INPUTS['crmp']['icrmp']['predict']['predict']
FIG_DIR = INPUTS['crmp']['figures_dir']

def net_production_vs_time():
    plt.figure()
    plt.plot(time, net_productions.T)
    plot_helper(
        FIG_DIR,
        xlabel='Time',
        ylabel='Net Production',
        legend=producer_names,
        save=True
    )


def net_production_estimators_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(metrics_file)
    x = np.arange(len(x_labels))
    width = 0.15
    bar_labels = [
        'ICRMP, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse',
        'Lasso, mse', 'Elastic, mse'
    ]
    for i in range(len(net_productions)):
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
        models = ['ICRMP', 'LinearRegression', 'BayesianRidge', 'Lasso', 'ElasticNet']
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
    predictions_metrics_df = pd.read_csv(metrics_file)
    x = np.arange(len(x_labels))
    width = 0.23
    bar_labels = [
        'ICRMP, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse'
    ]
    for i in range(len(net_productions)):
        producer = i + 1
        producer_rows_df = predictions_metrics_df.loc[predictions_metrics_df['Producer'] == producer]
        crmp_mse = []
        linear_regression_mse = []
        bayesian_ridge_mse = []
        heights = [
            crmp_mse, linear_regression_mse, bayesian_ridge_mse
        ]
        models = ['ICRMP', 'LinearRegression', 'BayesianRidge']
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


def net_production_with_predictions():
    fitting_df = pd.read_csv(fit_file)
    predictions_df = pd.read_csv(predict_file)
    for i in range(len(net_productions)):
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
        models = ['ICRMP', 'LinearRegression', 'BayesianRidge']
        # models = ['ICRMP', 'LinearRegression', 'BayesianRidge']
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


# net_production_vs_time()
# net_production_estimators_and_time_steps()
# net_production_good_estimators_and_time_steps()
net_production_with_predictions()
