import numpy as np

from src.data.read_crmp import injectors, producers, producer_names, true_params


step_sizes = np.linspace(2, 12, num=11).astype(int)
number_of_producers = len(producers)

number_of_gains = 6
# number_of_gains = 11
number_of_time_constants = 11
# number_of_time_constants = 201
f1 = np.linspace(0, 1, number_of_gains)
tau = np.linspace(0, 100, number_of_time_constants)
f2 = np.ones(number_of_gains) - f1
# TODO: I might be able to construct this using a meshgrid
param_grid = {'p0': []}
for i in tau:
    if i == 0:
        i = 1e-06
    for j in range(len(f1)):
        param_grid['p0'].append([i, f1[j], f2[j]])


injector_names = ['IA04', 'IA08', 'IA11', 'IA13']
producer_names = ['PA01', 'PA02', 'PA03', 'PA05', 'PA09', 'PA10', 'PA12']
