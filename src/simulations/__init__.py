import numpy as np

from src.data.read_crmp import injectors, producers, producer_names

step_sizes = np.linspace(2, 12, num=11).astype(int)

number_of_producers = len(producers)

f1 = np.linspace(0, 1, 6)
f2 = np.ones(6) - f1
tau = np.linspace(1e-6, 100, 10)
# TODO: I might be able to construct this using a meshgrid
param_grid = {'p0': []}
for i in tau:
    for j in range(len(f1)):
        param_grid['p0'].append([i, f1[j], f2[j]])
