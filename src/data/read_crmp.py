import numpy as np
import yaml

from src.config import INPUTS


def _insert_zero(column):
    column = np.insert(column, 0, 0., axis=0)
    return column


def read_crmp(data_file):
    data = np.loadtxt(data_file, delimiter=',', skiprows=1).T
    Time = data[0]
    Fixed_inj1 = data[1]
    Net_Fixed_inj1 = data[2]
    Fixed_inj2 = data[3]
    Net_Fixed_inj2 = data[4]
    q_1 = data[5]
    N_1 = data[6]
    q_2 = data[7]
    N_2 = data[8]
    q_3 = data[9]
    N_3 = data[10]
    q_4 = data[11]
    N_4 = data[12]
    features = [
        Time, Fixed_inj1, Net_Fixed_inj1, Fixed_inj2, Net_Fixed_inj2, q_1, N_1,
        q_2, N_2, q_3, N_3, q_4, N_4
    ]
    return features


data_file = INPUTS['crmp']['data']
features = read_crmp(data_file)
[
    Time, Fixed_inj1, Net_Fixed_inj1, Fixed_inj2, Net_Fixed_inj2, q_1, N_1,
    q_2, N_2, q_3, N_3, q_4, N_4
] = features
q_1 = np.insert(q_1, 0, 0., axis=0)
q_2 = np.insert(q_2, 0, 0., axis=0)
q_3 = np.insert(q_3, 0, 0., axis=0)
q_4 = np.insert(q_4, 0, 0., axis=0)

producers = np.array([q_1, q_2, q_3, q_4])
producer_names = [
    'Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'
]
injectors = np.array([Fixed_inj1, Fixed_inj2])
net_productions = np.array([
    N_1, N_2, N_3, N_4
])
N_1 = np.insert(N_1, 0, 0., axis=0)
N_2 = np.insert(N_2, 0, 0., axis=0)
N_3 = np.insert(N_3, 0, 0., axis=0)
N_4 = np.insert(N_4, 0, 0., axis=0)

actual_parameters = {
    1: [0.2, 0.8, 1.5],
    2: [0.4, 0.6, 1],
    3: [0.6, 0.4, 5],
    4: [0.8, 0.2, 50]
}

q_predictions_output_file = INPUTS['crmp']['q_predictions']
