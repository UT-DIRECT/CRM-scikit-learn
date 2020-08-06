import numpy as np
import yaml

from src.config import INPUTS
from src.helpers.features import net_flow


def read_wfsim(data_file):
    data = np.loadtxt(data_file, delimiter=',', skiprows=1).T
    time = data[0]
    delta_time = data[1]
    qo_tank = data[2]
    w_tank = data[3]
    qw_tank = data[4]
    q_tank = data[5]
    features = [time, delta_time, qo_tank, w_tank, qw_tank, q_tank]
    return features


data_file = INPUTS['wfsim']['data']
features = read_wfsim(data_file)

[time, delta_time, qo_tank, w_tank, qw_tank, q_tank] = features
f_w = qw_tank / q_tank
W_t = np.array(net_flow(w_tank))
Q_t = np.array(net_flow(q_tank))
Qo_t = np.array(net_flow(qo_tank))
Qw_t = np.array(net_flow(qw_tank))
