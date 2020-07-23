import numpy as np
import yaml

from src.config import INPUTS


def read_wfsim(data_file):
    pass


data_file = INPUTS['wfsim']['data']
features = read_wfsim(data_file)
[Time, qo_tank, w_tank, qw_tank, q_tank] = features
