import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (cross_val_score, GridSearchCV,
    train_test_split)

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import model_namer, test_model
from src.models.crmp import CRMP
from src.simulations import number_of_producers, param_grid


most_predictive_parameters_file = INPUTS['crmp']['crmp']['predict']['most_predictive_parameters']
most_predictive_parameters_data = {
    'Producer': [], 'tau_initial': [], 'tau_final': [], 'f1_initial': [],
    'f1_final': [], 'f2_initial': [], 'f2_final': [], 'r2': [], 'MSE': []
}

def train_crmp_across_wells():
    characteristic_producer = np.mean(producers, axis=0)
    X, y = production_rate_dataset(characteristic_producer, *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    for p0 in param_grid['p0']:
        crmp = CRMP(p0=p0)
        crmp = crmp.fit(X_train, y_train)
        mse = 0
        r2 = 0
        for i in range(number_of_producers):
            X_i, y_i = production_rate_dataset(producers[i], *injectors)
            X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
                X_i, y_i, test_size=0.2, shuffle=False
            )
            y_hat = crmp.predict(X_test_i)
            r2_i, mse_i = fit_statistics(y_hat, y_test_i)
            most_predictive_parameters_data['Producer'].append(i + 1)
            most_predictive_parameters_data['tau_initial'].append(p0[0])
            most_predictive_parameters_data['tau_final'].append(crmp.tau_)
            most_predictive_parameters_data['f1_initial'].append(p0[1])
            most_predictive_parameters_data['f1_final'].append(crmp.gains_[0])
            most_predictive_parameters_data['f2_initial'].append(p0[2])
            most_predictive_parameters_data['f2_final'].append(crmp.gains_[1])
            most_predictive_parameters_data['r2'].append(r2_i)
            most_predictive_parameters_data['MSE'].append(mse_i)

    most_predictive_parameters_df = pd.DataFrame(most_predictive_parameters_data)
    most_predictive_parameters_df.to_csv(most_predictive_parameters_file)

train_crmp_across_wells()
