import numpy as np
import yaml

from src.config import INPUTS


def _insert_zero(column):
    column = np.insert(column, 0, 0., axis=0)
    return column


def read_crmp(data_file):
    data = np.loadtxt(data_file, delimiter=',', skiprows=1).T
    time = data[0]
    fixed_inj1 = data[1]
    net_fixed_inj1 = data[2]
    fixed_inj2 = data[3]
    net_fixed_inj2 = data[4]
    q_1 = data[5]
    N_1 = data[6]
    q_2 = data[7]
    N_2 = data[8]
    q_3 = data[9]
    N_3 = data[10]
    q_4 = data[11]
    N_4 = data[12]
    features = [
        time, fixed_inj1, net_fixed_inj1, fixed_inj2, net_fixed_inj2, q_1, N_1,
        q_2, N_2, q_3, N_3, q_4, N_4
    ]
    return features


data_file = INPUTS['crmp']['data']
features = read_crmp(data_file)
[
    time, fixed_inj1, net_fixed_inj1, fixed_inj2, net_fixed_inj2, q_1, N_1,
    q_2, N_2, q_3, N_3, q_4, N_4
] = features

producers = np.array([q_1, q_2, q_3, q_4])
producers_new = []
for i in range(len(producers)):
    producers_new.append(_insert_zero(producers[i]))
producers = np.array(producers_new)

producer_names = [
    'Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'
]
injectors = np.array([fixed_inj1, fixed_inj2])
net_productions = np.array([N_1, N_2, N_3, N_4])
net_productions_new = []
for i in range(len(net_productions)):
    net_productions_new.append(_insert_zero(net_productions[i]))
net_productions = np.array(net_productions_new)

true_params = {
    1: [0.2, 1.5],
    2: [0.4, 1],
    3: [0.6, 5],
    4: [0.8, 50]
}
