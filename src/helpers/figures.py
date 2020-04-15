import matplotlib.pyplot as plt

from ..data.process_dataset import INPUTS


FIG_DIR = INPUTS['files']['figures_dir']


def plot_helper(title='', xlabel='', ylabel='', legend=[], save=False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    if save:
        fig_saver(title, xlabel, ylabel)
        plt.close()
    else:
        plt.show()


def fig_saver(title, xlabel, ylabel):
    fig_file = "{}{}".format(
        FIG_DIR,
        fig_filename(title, xlabel, ylabel)
    )
    plt.savefig(fig_file)


def fig_filename(title, xlabel, ylabel):
    return "{}_{}_{}.png".format(
        title,
        xlabel,
        ylabel
    ).lower().replace(' ', '_')
