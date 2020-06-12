import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import (injectors, net_productions,
    producers, producer_names, step_sizes, Time)
from src.helpers.figures import bar_plot_helper, bar_plot_formater, plot_helper
from src.visualization import I_predictions_file, net_I_predictions_file


I_predictions_df = pd.read_csv(I_predictions_file)
net_I_predictions_df = pd.read_csv(net_I_predictions_file)


def injection_rate_vs_time():
    for i in range(len(producers)):
        producer = i + 1
        producer_rows_df = I_predictions_df.loc[I_predictions_df['Producer'] == producer]
        t = producer_rows_df['t_i']
        print('producer_rows_df: ', producer_rows_df)
        print('t: ', t)
        injector_data = np.array([producer_rows_df['injector_1'], producer_rows_df['injector_1']]).T
        plt.figure()
        plt.plot(t, injector_data)
        plot_helper(
            title='Producer {}'.format(producer),
            xlabel='Time',
            ylabel='Injection Rate',
            legend=['Injector 1', 'Injector 2'],
            save=True
        )


def production_rate_vs_time():
    pass


def injection_production_rate_net_production_vs_time():
    pass


injection_rate_vs_time()
