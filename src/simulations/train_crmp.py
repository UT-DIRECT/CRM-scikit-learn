import pickle
import dill as pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import (BayesianRidge, ElasticNetCV, LassoCV,
        LinearRegression)
from sklearn.metrics import r2_score

from src.config import INPUTS
from src.data.read_crmp import (injectors, net_productions, producers,
         producer_names)
from src.helpers.cross_validation import (forward_walk_splitter,
        train_model_with_cv)
from src.helpers.features import (net_production_dataset,
        production_rate_dataset)
from src.helpers.models import model_namer, serialized_model_path, is_CV_model
from src.models.crmp import CRMP
from src.models.net_crm import NetCRM


# Production Rate Training
q_fitting_file = INPUTS['crmp']['q_fitting']
q_fitting_data = {
    'Producer': [], 'Model': [], 't_i': [], 'Fit': []
}
for i in range(len(producers)):
    models = [
        BayesianRidge(), CRMP(), ElasticNetCV, LassoCV, LinearRegression()
    ]
    X, y = production_rate_dataset(producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(X, y, 2)
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    for model in models:
        if is_CV_model(model):
            model = train_model_with_cv(X, y, model, train_split)
        model = model.fit(X_train, y_train)
        y_hat = model.predict(X_train)
        time = np.linspace(1, len(y_hat), num=len(y_hat))
        # TODO: This is not the ideal location for getting this fitting data.
        for k in range(len(y_hat)):
            q_fitting_data['Producer'].append(i + 1)
            q_fitting_data['Model'].append(model_namer(model))
            q_fitting_data['t_i'].append(k + 1)
            q_fitting_data['Fit'].append(y_hat[k])
        pickled_model = serialized_model_path('crmp', model, producer_names[i])
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)

q_fitting_df = pd.DataFrame(q_fitting_data)
q_fitting_df.to_csv(q_fitting_file)


# Net Production Training
N_fitting_file = INPUTS['crmp']['N_fitting']
N_fitting_data = {
    'Producer': [], 'Model': [], 't_i': [], 'Fit': []
}
for i in range(len(producers)):
    models = [
        BayesianRidge(), NetCRM(), ElasticNetCV, LassoCV, LinearRegression()
    ]
    X, y = net_production_dataset(net_productions[i], producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(X, y, 2)
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    for model in models:
        if is_CV_model(model):
            model = train_model_with_cv(X, y, model, train_split)
        model = model.fit(X_train, y_train)
        y_hat = model.predict(X_train)
        time = np.linspace(1, len(y_hat), num=len(y_hat))
        for k in range(len(y_hat)):
            N_fitting_data['Producer'].append(i + 1)
            N_fitting_data['Model'].append(model_namer(model))
            N_fitting_data['t_i'].append(k + 1)
            N_fitting_data['Fit'].append(y_hat[k])
        pickled_model = serialized_model_path(
            'net_crm', model, 'Net {}'.format(producer_names[i])
        )
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)

N_fitting_df = pd.DataFrame(N_fitting_data)
N_fitting_df.to_csv(N_fitting_file)
