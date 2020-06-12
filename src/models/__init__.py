import numpy as np
import yaml

from src.config import INPUTS


def main():
    def read_data( data_file):
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


    data_file = INPUTS['files']['data']
    features = read_data(data_file)
    [
        Time, Fixed_inj1, Net_Fixed_inj1, Fixed_inj2, Net_Fixed_inj2, q_1, N_1,
        q_2, N_2, q_3, N_3, q_4, N_4
    ] = features

    producers = np.array([q_1, q_2, q_3, q_4])
    producer_names = [
        'Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'
    ]
    injectors = np.array([Fixed_inj1, Fixed_inj2])
    net_productions = np.array([
        N_1, N_2, N_3, N_4
    ])

    step_sizes = np.linspace(2, 12, num=11).astype(int)
    q_predictions_output_file = INPUTS['files']['q_predictions']


if __name__ == 'main':
    main()
