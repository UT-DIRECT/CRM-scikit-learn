import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import model_namer, test_model
from src.models.crmp import CRMP
from src.simulations import number_of_producers, param_grid


fit_ouput_file = INPUTS['crmp']['crmp']['fit']['sensitivity_analysis']
predict_output_file = INPUTS['crmp']['crmp']['predict']['sensitivity_analysis']
fit_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'r2': [], 'MSE': []
}
predict_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'r2': [], 'MSE': []
}

for i in range(number_of_producers):
    X, y = production_rate_dataset(producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
        X, y, 2, training_split=0.5
    )
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]
    X_test = X[train_test_seperation_idx:]
    y_test = y[train_test_seperation_idx:]
    for p0 in param_grid['p0']:
        crmp = CRMP(p0=p0)
        crmp = crmp.fit(X_train, y_train)

        # Fitting
        y_hat = crmp.predict(X_train)
        r2, mse = fit_statistics(y_hat, y_train)
        fit_data['Producer'].append(i + 1)
        fit_data['Model'].append(model_namer(crmp))
        fit_data['tau_initial'].append(p0[0])
        fit_data['tau_final'].append(crmp.tau_)
        fit_data['f1_initial'].append(p0[1])
        fit_data['f1_final'].append(crmp.gains_[0])
        fit_data['f2_initial'].append(p0[2])
        fit_data['f2_final'].append(crmp.gains_[1])
        fit_data['r2'].append(r2)
        fit_data['MSE'].append(mse)

        # Prediction
        y_hat = crmp.predict(X_test)
        r2, mse = fit_statistics(y_hat, y_test)
        predict_data['Producer'].append(i + 1)
        predict_data['Model'].append(model_namer(crmp))
        predict_data['tau_initial'].append(p0[0])
        predict_data['tau_final'].append(crmp.tau_)
        predict_data['f1_initial'].append(p0[1])
        predict_data['f1_final'].append(crmp.gains_[0])
        predict_data['f2_initial'].append(p0[2])
        predict_data['f2_final'].append(crmp.gains_[1])
        predict_data['r2'].append(r2)
        predict_data['MSE'].append(mse)

# Fitting
fit_df = pd.DataFrame(fit_data)
fit_df.to_csv(fit_ouput_file)

# Prediction
predict_df = pd.DataFrame(predict_data)
predict_df.to_csv(predict_output_file)
