import pickle
import dill as pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml

from lmfit import Model, Parameters
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
        LinearRegression)

from src.config import INPUTS
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import serialized_model_file
from src.models.crm import CRM


# TODO: make a CRM net production version
def N2(self, N1, q2):
    return N1 + q2


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
net_productions = np.array([
    N_1, N_2, N_3, N_4
])

step_sizes = np.linspace(2, 12, num=11).astype(int)
N_predictions_output_file = INPUTS['files']['N_predictions']

for i in range(len(producers)):
    models = [
        BayesianRidge(), CRM(), ElasticNet(), Lasso(), LinearRegression()
    ]
    X, y = production_rate_dataset(producers[i], Fixed_inj1, Fixed_inj2)
    train_test_seperation_idx = forward_walk_splitter(X, y, 2)[2]
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    for model in models:
        model = model.fit(X_train, y_train)
        pickled_model = serialized_model_file(producer_names[i], model)
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)
