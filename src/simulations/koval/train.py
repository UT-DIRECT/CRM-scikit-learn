import pickle
import dill as pickle
import numpy as np
import pandas as pd

from crmp import Koval
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, LassoCV,
        LinearRegression)
from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import (delta_time, f_w, q_tank, qo_tank, qw_tank,
         Q_t, Qo_t, Qw_t, time, w_tank, W_t)
from src.helpers.cross_validation import (forward_walk_splitter,
        train_model_with_cv)
from src.helpers.features import koval_dataset, production_rate_dataset
from src.helpers.models import model_namer, serialized_model_path, is_CV_model

koval_fitting_file = INPUTS['wfsim']['koval_fitting']
koval_fitting_data = {
    'Model': [], 't_i': [], 'Fit': []
}
X, y = koval_dataset(W_t, f_w)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1, shuffle=False
)
koval = Koval().fit(X=X_train, y=y_train)
y_hat = koval.predict(X_train)
time = np.linspace(1, len(y_hat), num=len(y_hat))
for k in range(len(y_hat)):
    koval_fitting_data['Model'].append(model_namer(koval))
    koval_fitting_data['t_i'].append(k + 1)
    koval_fitting_data['Fit'].append(y_hat[k])

pickled_model = serialized_model_path('koval', koval)
with open(pickled_model, 'wb') as f:
    pickle.dump(koval, f)

X, y = production_rate_dataset(f_w, W_t)
train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
    X, y, 2
)
X_train = X[:train_test_seperation_idx]
y_train = y[:train_test_seperation_idx]
models = [
    BayesianRidge(), ElasticNetCV, LassoCV, LinearRegression()
]
for model in models:
    if is_CV_model(model):
        model = train_model_with_cv(X, y, model, train_split)
    model = model.fit(X_train, y_train)
    y_hat = model.predict(X_train)
    time = np.linspace(1, len(y_hat), num=len(y_hat))
    for k in range(len(y_hat)):
        koval_fitting_data['Model'].append(model_namer(model))
        koval_fitting_data['t_i'].append(k + 1)
        koval_fitting_data['Fit'].append(y_hat[k])
    pickled_model = serialized_model_path('koval', model)
    with open(pickled_model, 'wb') as f:
        pickle.dump(model, f)

koval_fitting_df = pd.DataFrame(koval_fitting_data)
koval_fitting_df.to_csv(koval_fitting_file)
