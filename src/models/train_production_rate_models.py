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
from src.helpers.features import net_production_dataset, production_rate_dataset
from src.helpers.models import serialized_model_path
from src.models import injectors, net_productions, producers, producer_names
from src.models.crm import CRM


def is_CV_model(model):
    return not (isinstance(model, LinearRegression) or isinstance(model, BayesianRidge) or isinstance(model, CRM))


# Production Rate Training
for i in range(len(producers)):
    models = [
        BayesianRidge(), CRM(), ElasticNetCV, LassoCV, LinearRegression()
    ]
    X, y = production_rate_dataset(producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(X, y, 2)
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    for model in models:
        if is_CV_model(model):
            model = train_model_with_cv(X, y, model, train_split)
        model = model.fit(X_train, y_train)
        pickled_model = serialized_model_path(producer_names[i], model)
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)


# Net Production Training
for i in range(len(producers)):
    models = [
        BayesianRidge(), CRM(), ElasticNetCV, LassoCV, LinearRegression()
    ]
    X, y = net_production_dataset(net_productions[i], producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(X, y, 2)
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    for model in models:
        if is_CV_model(model):
            model = train_model_with_cv(X, y, model, train_split)
        model = model.fit(X_train, y_train)
        pickled_model = serialized_model_path(
            'Net {}'.format(producer_names[i]),
            model
        )
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)
