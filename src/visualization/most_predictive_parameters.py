import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.read_crmp import producers
from src.helpers.figures import plot_helper
from src.visualization import INPUTS


most_predictive_parameters_file = INPUTS['crmp']['crmp']['predict']['most_predictive_parameters']
most_predictive_parameters_df = pd.read_csv(q_fitting_sensitivity_analysis_file)

FIG_DIR = INPUTS['crmp']['figures_dir']

number_of_time_constants = 10
number_of_gains = 6
xlabel ='f1'
ylabel ='tau'


def most_predictive_parameters_contour_map():
    x = df['f1_initial'].to_numpy()
    x = np.reshape(x, (number_of_time_constants, number_of_gains))
    y = df['tau_initial'].to_numpy()
    y = np.reshape(y, (number_of_time_constants, number_of_gains))
    z = df['MSE'].to_numpy()
    z_tmp = []
    for i in z:
        if i == 0:
            z_tmp.append(i)
        else:
            z_tmp.append(np.log(i))
    z = z_tmp
    z = np.reshape(z, (number_of_time_constants, number_of_gains))
    return (x, y, z)


most_predictive_parameters_contour_map()
