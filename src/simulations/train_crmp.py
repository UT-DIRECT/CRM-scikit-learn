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
from src.models.icrmp import ICRMP


# Production Rate Training
output_file = INPUTS['crmp']['crmp']['fit']['fit']
data = {
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
            data['Producer'].append(i + 1)
            data['Model'].append(model_namer(model))
            data['t_i'].append(k + 1)
            data['Fit'].append(y_hat[k])
        pickled_model = serialized_model_path('crmp', model, producer_names[i])
        with open(pickled_model, 'wb') as f:
            pickle.dump(model, f)

fit = pd.DataFrame(data)
fit.to_csv(output_file)
