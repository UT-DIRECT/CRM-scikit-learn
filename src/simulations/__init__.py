import numpy as np

from src.data.read_crmp import injectors, producers, producer_names

step_sizes = np.linspace(2, 12, num=11).astype(int)

number_of_producers = len(producers)


number_of_gains = 6
number_of_time_constants = 10
f1 = np.linspace(0, 1, number_of_gains)
f2 = np.ones(number_of_gains) - f1
tau = np.linspace(1e-6, 100, number_of_time_constants)
# TODO: I might be able to construct this using a meshgrid
param_grid = {'p0': []}
for i in tau:
    for j in range(len(f1)):
        param_grid['p0'].append([i, f1[j], f2[j]])
