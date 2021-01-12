import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import INPUTS
from src.data.read_crmp import (injectors, net_productions, producers,
         producer_names)
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import net_production_dataset
from src.helpers.models import model_namer
from src.models.icrmp import ICRMP
from src.simulations import number_of_producers, param_grid


fit_file = INPUTS['crmp']['icrmp']['fit']['sensitivity_analysis']
fit_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'r2': [], 'MSE': []
}

for i in range(number_of_producers):
    X, y = net_production_dataset(net_productions[i], producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
        X, y, 2
    )
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]
    X_test = X[train_test_seperation_idx:]
    y_test = y[train_test_seperation_idx:]
    for p0 in param_grid['p0']:
        icrmp = ICRMP(p0=p0)
        icrmp = icrmp.fit(X_train, y_train)
        y_hat = icrmp.predict(X_train)
        r2, mse = fit_statistics(y_train, y_hat)
        fit_data['Producer'].append(i + 1)
        fit_data['Model'].append(model_namer(icrmp))
        fit_data['tau_initial'].append(p0[0])
        fit_data['tau_final'].append(icrmp.tau_)
        fit_data['f1_initial'].append(p0[1])
        fit_data['f1_final'].append(icrmp.gains_[0])
        fit_data['f2_initial'].append(p0[2])
        fit_data['f2_final'].append(icrmp.gains_[1])
        fit_data['r2'].append(r2)
        fit_data['MSE'].append(mse)

fit_df = pd.DataFrame(fit_data)
fit_df.to_csv(output_file)
