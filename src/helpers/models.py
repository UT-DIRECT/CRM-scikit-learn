from os import listdir
from os.path import isfile, join

import pickle
import dill as pickle

from src.helpers.analysis import fit_statistics


TRAINED_MODEL_DIR = './models'


def serialized_model_path(producer_name, model):
    model_name = model_namer(model)
    return '{}/{}_{}.pkl'.format(
        TRAINED_MODEL_DIR,
        producer_name,
        model_name
    ).lower().replace(' ', '_')


def model_namer(model):
    # Removes the parameters found in the model name
    return str(model)[:str(model).index('(')]


def load_models():
    dir_contents = listdir(TRAINED_MODEL_DIR)
    dir_files = []
    for f in dir_contents:
        if isfile(join(TRAINED_MODEL_DIR, f)) and f != '.gitkeep':
            dir_files.append(f)
    models = {}
    for f in dir_files:
        models[f[:-4]] = _load_model(f)
    return models


def _load_model(f):
    with open(join(TRAINED_MODEL_DIR, f), 'rb') as f:
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
