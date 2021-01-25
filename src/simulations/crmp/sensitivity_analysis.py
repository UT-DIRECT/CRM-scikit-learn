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
fit_df = pd.DataFrame(
    columns=[
        'Producer', 'Model', 'tau_initial', 'tau_final',
        'f1_initial', 'f1_final', 'f2_initial', 'f2_final',
        'r2', 'MSE'
    ]
)
predict_df = pd.DataFrame(
    columns=[
        'Producer', 'Model', 'tau_initial', 'tau_final',
        'f1_initial', 'f1_final', 'f2_initial', 'f2_final',
        'r2', 'MSE'
    ]
)

for i in range(number_of_producers):
    producer_number = i + 1
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
        fit_df.loc[len(fit_df.index)] = [
            producer_number, model_namer(crmp), p0[0], crmp.tau_, p0[1],
            crmp.gains_[0], p0[2], crmp.gains_[1], r2, mse
        ]

        # Prediction
        y_hat = crmp.predict(X_test)
        r2, mse = fit_statistics(y_hat, y_test)
        predict_df.loc[len(predict_df.index)] = [
            producer_number, model_namer(crmp), p0[0], crmp.tau_, p0[1],
            crmp.gains_[0], p0[2], crmp.gains_[1], r2, mse
        ]

# Fitting
fit_df.to_csv(fit_ouput_file)

# Prediction
predict_df.to_csv(predict_output_file)
