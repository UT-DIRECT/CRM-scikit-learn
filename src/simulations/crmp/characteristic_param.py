import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (cross_val_score, GridSearchCV,
    train_test_split)

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, Time
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import model_namer, test_model
from src.models.crmp import CRMP
from src.simulations import number_of_producers, param_grid


characteristic_params_file = INPUTS['crmp']['crmp']['predict']['characteristic_params']
characteristic_params_predictions_file = INPUTS['crmp']['crmp']['predict']['characteristic_params_predictions']
characteristic_objective_function_file = INPUTS['crmp']['crmp']['predict']['characteristic_objective_function']

characteristic_params_data = {
    'Producer': [], 'tau_initial': [], 'tau_final': [], 'f1_initial': [],
    'f1_final': [], 'f2_initial': [], 'f2_final': [], 'r2': [], 'MSE': []
}
characteristic_params_predictions_data = {
    'Producer': np.array([]), 't_i': np.array([]), 'Prediction': np.array([])
}
characteristic_objective_function_data = {
    'tau': [], 'f1': [], 'f2': [], 'r2': [], 'MSE': []
}

def train_crmp_across_wells():
    characteristic_producer = np.mean(producers, axis=0)
    X, y = production_rate_dataset(characteristic_producer, *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    t_range = Time[len(X_train):]
    for p0 in param_grid['p0']:
        crmp = CRMP(p0=p0)
        crmp = crmp.fit(X_train, y_train)
        for i in range(number_of_producers):
            producer = i + 1
            X_i, y_i = production_rate_dataset(producers[i], *injectors)
            X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
                X_i, y_i, test_size=0.2, shuffle=False
            )
            y_hat = crmp.predict(X_test_i)
            r2_i, mse_i = fit_statistics(y_hat, y_test_i)
            characteristic_params_data['Producer'].append(producer)
            characteristic_params_data['tau_initial'].append(p0[0])
            characteristic_params_data['tau_final'].append(crmp.tau_)
            characteristic_params_data['f1_initial'].append(p0[1])
            characteristic_params_data['f1_final'].append(crmp.gains_[0])
            characteristic_params_data['f2_initial'].append(p0[2])
            characteristic_params_data['f2_final'].append(crmp.gains_[1])
            characteristic_params_data['r2'].append(r2_i)
            characteristic_params_data['MSE'].append(mse_i)
    characteristic_params_df = pd.DataFrame(
        characteristic_params_data
    )
    characteristic_params_df.to_csv(characteristic_params_file)


def characteristic_objective_function():
    number_of_gains = 20
    number_of_time_constants = 100
    f1 = np.linspace(0, 1, number_of_gains)
    f2 = np.ones(number_of_gains) - f1
    tau = np.linspace(1e-6, 100, number_of_time_constants)
    # TODO: I might be able to construct this using a meshgrid
    param_grid = {'p0': []}
    for i in tau:
        for j in range(len(f1)):
            param_grid['p0'].append([i, f1[j], f2[j]])
    crmp = CRMP()
    X, y = production_rate_dataset(producers[0], *injectors)
    crmp = crmp.fit(X, y)
    for p0 in param_grid['p0']:
        crmp.tau_ = p0[0]
        crmp.gains_ = p0[1:]
        r2 = 0
        mse = 0
        for i in range(number_of_producers):
            producer = i + 1
            X, y = production_rate_dataset(producers[i], *injectors)
            X_train, X_test, y_train, y_test= train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            y_hat = crmp.predict(X_train)
            r2_i, mse_i = fit_statistics(y_hat, y_train)
            r2 += r2_i
            mse += mse_i
        r2 /= 4
        mse /= 4
        characteristic_objective_function_data['tau'].append(p0[0])
        characteristic_objective_function_data['f1'].append(p0[1])
        characteristic_objective_function_data['f2'].append(p0[2])
        characteristic_objective_function_data['r2'].append(r2)
        characteristic_objective_function_data['MSE'].append(mse)
    characteristic_objective_function_df = pd.DataFrame(
        characteristic_objective_function_data
    )
    characteristic_objective_function_df.to_csv(
        characteristic_objective_function_file
    )

# train_crmp_across_wells()
characteristic_objective_function()
