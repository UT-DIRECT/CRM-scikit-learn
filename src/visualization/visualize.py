import matplotlib.pyplot as plt

from src.models import (injectors, net_productions, producers, producer_names,
    Time)
from src.helpers.figures import plot_helper


def producers_vs_time():
    plt.figure()
    plt.plot(Time, producers.T)
    plot_helper(
        title='Production Rate vs Time',
        xlabel='Time',
        ylabel='Production Rate',
        legend=producer_names,
        save=True
    )


def net_production_vs_time():
    plt.figure()
    plt.plot(Time, net_productions.T)
    plot_helper(
        title='Total Production vs Time',
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
            title='Injector {}'.format(i + 1),
            xlabel='Injection Rate',
            ylabel='Production Rate',
            legend=producer_names,
            save=True
        )


def net_production_estimators_and_time_steps():
    pass


def net_production_good_estimators_and_time_steps():
    pass


def animated_net_production_predictions():
    pass


producers_vs_time()
net_production_vs_time()
producers_vs_injector()
