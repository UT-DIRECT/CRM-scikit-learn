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


q_sensitivity_analysis_file = INPUTS['crmp']['q_sensitivity_analysis']
q_sensitivity_analysis_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'r2': [], 'MSE': []
}

f1 = np.linspace(0, 1, 11)
f2 = np.ones(11) - f1
tau = np.linspace(1, 100, 100)
# TODO: I might be able to construct this using a meshgrid
param_grid = {'p0': []}
for i in tau:
    for j in range(len(f1)):
        param_grid['p0'].append([i, f1[j], f2[j]])

for i in range(len(producers)):
    X, y = production_rate_dataset(producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
        X, y, 2
    )
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]
    X_test = X[train_test_seperation_idx:]
    y_test = y[train_test_seperation_idx:]
#     crmp_cv = GridSearchCV(CRMP(), param_grid,
#         scoring=make_scorer(mean_squared_error, greater_is_better=False),
#         cv=train_split
#     )
#     crmp_cv.fit(X, y)
#     print(crmp_cv.get_params())
#     raise
#     print(crmp_cv.best_params_)
#     print(crmp_cv.best_score_)
#     print(1)
#     print(crmp_cv.predict(X_test))
#     print(cross_val_score(crmp_cv, X, y, cv=test_split))
#     print(2)
    for p0 in param_grid['p0']:
        crmp = CRMP(p0=p0)
        crmp = crmp.fit(X_train, y_train)
        r2, mse, y_hat, time_step = test_model(X, y, crmp, test_split)
        # y_hat = crmp.predict(X_train)
        # r2, mse = fit_statistics(y_train, y_hat)
        q_sensitivity_analysis_data['Producer'].append(i + 1)
        q_sensitivity_analysis_data['Model'].append(model_namer(crmp))
        q_sensitivity_analysis_data['tau_initial'].append(p0[0])
        q_sensitivity_analysis_data['tau_final'].append(crmp.tau_)
        q_sensitivity_analysis_data['f1_initial'].append(p0[1])
        q_sensitivity_analysis_data['f1_final'].append(crmp.gains_[0])
        q_sensitivity_analysis_data['f2_initial'].append(p0[2])
        q_sensitivity_analysis_data['f2_final'].append(crmp.gains_[1])
        q_sensitivity_analysis_data['r2'].append(r2)
        q_sensitivity_analysis_data['MSE'].append(mse)

q_sensitivity_analysis_df = pd.DataFrame(q_sensitivity_analysis_data)
q_sensitivity_analysis_df.to_csv(q_sensitivity_analysis_file)
