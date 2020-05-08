import pickle
import dill as pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, LassoCV,
        LinearRegression)

from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import (forward_walk_splitter,
        train_model_with_cv)
from src.helpers.features import production_rate_dataset
from src.helpers.models import serialized_model_file
from src.models import producers, Fixed_inj1, Fixed_inj2, producer_names
from src.models.crm import CRM


# TODO: make a CRM net production version
def N2(self, N1, q2):
    return N1 + q2


def is_CV_model(model):
    return not (isinstance(model, LinearRegression) or isinstance(model, BayesianRidge) or isinstance(model, CRM))


for i in range(len(producers)):
    models = [
        BayesianRidge(), CRM(), ElasticNetCV, LassoCV, LinearRegression()
    ]
    X, y = production_rate_dataset(producers[i], Fixed_inj1, Fixed_inj2)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(X, y, 2)
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    for model in models:
        if is_CV_model(model):
            model = train_model_with_cv(X, y, model, train_split)
        model = model.fit(X_train, y_train)
        pickled_model = serialized_model_file(producer_names[i], model)
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)
