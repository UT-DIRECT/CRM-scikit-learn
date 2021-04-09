import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import INPUTS
from src.data.read_clair import (
    injectors, producers, producer_names, producer_starting_indicies, time
)
from src.helpers.figures import plot_helper

FIG_DIR = INPUTS['real_data']['figures_dir']
fit_file = INPUTS['real_data']['fit']['sensitivity_analysis']
fit_df = pd.read_csv(fit_file)

predict_file = INPUTS['real_data']['predict']['sensitivity_analysis']
predict_df = pd.read_csv(predict_file)


def plot_production_rate():
    tmp_producer_names = ['PA09', 'PA12']
    for name in tmp_producer_names:
        i = producer_names.index(name)
        print(i)
        producer = producers[i]
        starting_index = producer_starting_indicies[i]
        plt.plot(time[starting_index:], producer[starting_index:])
    plot_helper(
        FIG_DIR,
        xlabel='Date',
        ylabel='Production Rate',
        legend=tmp_producer_names,
        save=True
    )


plot_production_rate()
