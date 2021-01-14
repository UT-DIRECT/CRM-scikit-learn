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
fit_df = pd.DataFrame(
    columns=[
        'Producer', 'Model', 'tau_initial', 'tau_final',
        'f1_initial', 'f1_final', 'f2_initial', 'f2_final',
        'r2', 'MSE'
    ]
)

for i in range(number_of_producers):
    producer_number = i + 1
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

        # Fitting
        y_hat = icrmp.predict(X_train)
        r2, mse = fit_statistics(y_train, y_hat)
        fit_df.loc[len(fit_df.index)] = [
            producer_number, model_namer(icrmp), p0[0], icrmp.tau_, p0[1],
            icrmp.gains_[0], p0[2], icrmp.gains_[1], r2, mse
        ]

fit_df.to_csv(fit_file)
