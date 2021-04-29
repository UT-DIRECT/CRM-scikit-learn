import matplotlib.pyplot as plt
import numpy as np

from src.config import INPUTS
from src.simulations import (
    number_of_gains, number_of_time_constants
)


def plot_helper(fig_dir, title='', xlabel='', ylabel='', legend=None, save=False):
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if legend is not None:
        if legend is True:
            plt.legend()
        else:
            plt.legend(legend)
    if save:
        fig_saver(fig_dir, title, xlabel, ylabel)
        plt.close()


def fig_saver(fig_dir, title, xlabel, ylabel):
    fig_file = "{}{}".format(
        fig_dir,
        fig_filename(title, xlabel, ylabel)
    )
    plt.savefig(fig_file, bbox_inches='tight')


def fig_filename(title, xlabel, ylabel):
    filename = ''
    if len(title) > 0:
        filename = '{}_{}_{}.png'.format(
            title,
            xlabel,
            ylabel
        ).lower().replace(' ', '_')
    else:
        filename = '{}_{}.png'.format(
            xlabel,
            ylabel
        ).lower().replace(' ', '_')
    return filename


def bar_plot_helper(width, x, x_labels, bar_labels, heights):
    plt.figure(figsize=[10, 4.8])
    center_x_location = int(len(heights) / 2)
    for i in range(len(heights)):
        if i == 0:
            alpha = 1
        else:
            alpha = 0.5
        plt.bar(x + (i - center_x_location) * width, heights[i], width, label=bar_labels[i], alpha=alpha)


def bar_plot_formater(fig_dir, x, x_labels, title, xlabel, ylabel, legend=True):
    plot_helper(fig_dir, title=title, xlabel=xlabel, ylabel=ylabel)
    plt.yscale('log')
    plt.xticks(ticks=x, labels=x_labels)
    if legend:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    fig_saver(fig_dir, title, xlabel, ylabel)
    plt.close()


def initial_and_final_params_from_df(df):
    x_i = df['f1_initial']
    x_f = df['f1_final']
    y_i = df['tau_initial']
    y_f = df['tau_final']
    x = np.array([x_i, x_f]).T
    y = np.array([y_i, y_f]).T
    return (x, y)


def contour_params(df, x_column='', y_column='', z_column='',
                    shape=(number_of_time_constants, number_of_gains)):
    x = df[x_column].to_numpy()
    x = np.reshape(x, shape)
    y = df[y_column].to_numpy()
    y = np.reshape(y, shape)
    z = df[z_column].to_numpy()
    z_tmp = []
    for i in z:
        if i < 1e-06:
            z_tmp.append(-20)
        else:
            z_tmp.append(np.log(i))
    z = z_tmp
    z = np.reshape(z, shape)
    return (x, y, z)
