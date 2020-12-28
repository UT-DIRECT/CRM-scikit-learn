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


most_predictive_parameters_file = INPUTS['crmp']['crmp']['predict']['most_predictive_parameters']
most_predictive_parameters_data = {
    'tau': [], 'f1': [], 'f2': [], 'q1_mse': [], 'q2_mse': [], 'q3_mse': [],
    'q4_mse': [], 'aggregate_mses': []
}

f1 = np.linspace(0, 1, 6)
f2 = np.ones(6) - f1
tau = np.linspace(1e-6, 100, 10)
# TODO: I might be able to construct this using a meshgrid
param_grid = {'p0': []}
for i in tau:
    for j in range(len(f1)):
        param_grid['p0'].append([i, f1[j], f2[j]])


def predictive_value_of_different_parameters():
    X, y = production_rate_dataset(producers[0], *injectors)
    crmp = CRMP().fit(X, y)
    for p0 in param_grid['p0']:
        tau = p0[0]
        gains = [p0[1], p0[2]]
        crmp.tau_ = tau
        crmp.gains_ = gains
        mses = 0

        for i in range(len(producers)):
            X, y = production_rate_dataset(producers[i], *injectors)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            y_hat = crmp.predict(X_test)
            r2, mse = fit_statistics(y_hat, y_test)
            most_predictive_parameters_data['q{}_mse'.format(i + 1)].append(mse)
            mses += mse

        most_predictive_parameters_data['tau'].append(tau)
        most_predictive_parameters_data['f1'].append(gains[0])
        most_predictive_parameters_data['f2'].append(gains[1])
        most_predictive_parameters_data['aggregate_mses'].append(mses)


predictive_value_of_different_parameters()
most_predictive_parameters_df = pd.DataFrame(most_predictive_parameters_data)
most_predictive_parameters_df.to_csv(most_predictive_parameters_file)
