import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_wfsim import f_w, W_t
from src.helpers.figures import bar_plot_formater, plot_helper
from src.simulations import step_sizes
from src.visualization import INPUTS


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


def total_water_injected_and_predicted_water_cut():
    pass


def koval_estimation_error_and_time_steps():
    x_labels = [int(step_size) for step_size in step_sizes]
    predictions_metrics_df = pd.read_csv(koval_predictions_metrics_file)
    x = np.arange(len(x_labels))
    width = 0.25
    heights = []
    for i in range(len(step_sizes)):
        height = predictions_metrics_df.loc[predictions_metrics_df['Step size'] == step_sizes[i]]['MSE']
        heights.append(float(height))
    plt.bar(x, heights, width, label='CRMT, mse', alpha=0.5)
    title = 'Koval Fractional Flow Estimators'
    xlabel = 'Step Size'
    ylabel = 'Mean Squared Error'
    bar_plot_formater(FIG_DIR, x, x_labels, title, xlabel, ylabel, legend=False)


total_water_injected_and_water_cut()
koval_estimation_error_and_time_steps()
