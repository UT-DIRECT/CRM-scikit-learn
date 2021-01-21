import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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
predict_file = INPUTS['crmp']['icrmp']['predict']['sensitivity_analysis']
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
    X, y = net_production_dataset(net_productions[i], producers[i], *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    for p0 in param_grid['p0']:
        icrmp = ICRMP(p0=p0)
        icrmp = icrmp.fit(X_train, y_train)

        # Fitting
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

        # Predict
        y_hat = icrmp.predict(X_test)
        r2, mse = fit_statistics(y_test, y_hat)
        predict_data['Producer'].append(i + 1)
        predict_data['Model'].append(model_namer(icrmp))
        predict_data['tau_initial'].append(p0[0])
        predict_data['tau_final'].append(icrmp.tau_)
        predict_data['f1_initial'].append(p0[1])
        predict_data['f1_final'].append(icrmp.gains_[0])
        predict_data['f2_initial'].append(p0[2])
        predict_data['f2_final'].append(icrmp.gains_[1])
        predict_data['r2'].append(r2)
        predict_data['MSE'].append(mse)

fit_df = pd.DataFrame(fit_data)
fit_df.to_csv(fit_file)

predict_df = pd.DataFrame(predict_data)
predict_df.to_csv(predict_file)
