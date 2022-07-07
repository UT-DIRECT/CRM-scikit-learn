from abc import ABCMeta
from os import listdir
from os.path import isfile, join

import pickle
import dill as pickle

from sklearn.linear_model import BayesianRidge, LinearRegression

from src.helpers.analysis import fit_statistics


TRAINED_MODEL_DIR = './models'


def serialized_model_path(subdir, model, producer_name=''):
    model_name = model_namer(model)
    if len(producer_name) > 0:
        path = '{}/{}/{}_{}.pkl'.format(
            TRAINED_MODEL_DIR, subdir, producer_name, model_name
        )
    else:
        path = '{}/{}/{}.pkl'.format(
            TRAINED_MODEL_DIR, subdir, model_name
        )
    return path.lower().replace(' ', '_')


def model_namer(model):
    # Removes the parameters found in the model name
    return str(model)[:str(model).index('(')]


def load_models(subdir=''):
    dir_name = '{}/{}'.format(TRAINED_MODEL_DIR, subdir)
    dir_contents = listdir(dir_name)
    dir_files = []
    for f in dir_contents:
        if isfile(join(dir_name, f)) and f != '.gitkeep':
            dir_files.append(f)
    models = {}
    for f in dir_files:
        models[f[:-4]] = _load_model(dir_name, f)
    return models


def _load_model(dir_name, f):
    with open(join(dir_name, f), 'rb') as f:
        model = pickle.load(f)
    return model


def test_model(X, y, model, test_split):
    r2_sum, mse_sum = 0, 0
    length = len(test_split)
    y_hat = []
    time_step = []
    for train, test in test_split:
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model.fit(x_train, y_train)
        y_hat_i = model.predict(x_test)
        y_hat.append(y_hat_i)
        time_step.append(test)
        r2_i, mse_i = fit_statistics(y_hat_i, y_test)
        r2_sum += r2_i
        mse_sum += mse_i
    r2 = r2_sum / length
    mse = mse_sum / length
    return (r2, mse, y_hat, time_step)


def is_CV_model(model):
    if type(model) == ABCMeta:
        return True
    return False
