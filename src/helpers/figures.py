import matplotlib.pyplot as plt

FIG_DIR = "/Users/akhilpotla/ut/research/crm_validation/reports/figures/"

def plot_helper(title='', xlabel='', ylabel='', legend=[], save=False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    if save:
        fig_file = "{}{}".format(
            FIG_DIR,
            fig_filename(title, xlabel, ylabel)
        )
        plt.savefig(fig_file)
        plt.close()
    else:
        plt.show()

def fig_filename(title, xlabel, ylabel):
    return "{}_{}_{}.png".format(
        title,
        xlabel,
        ylabel
    ).lower().replace(' ', '_')

