import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.features import (
    get_real_producer_data, impute_training_data, production_rate_dataset,
    producer_rows_from_df, construct_real_production_rate_dataset
)
from src.helpers.cross_validation import scorer
from src.helpers.models import model_namer
from src.models.crmp import CRMP
from src.models.MBBaggingRegressor import MBBaggingRegressor
from src.simulations import injector_names, producer_names


injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])


def train_bagging_regressor_with_crmp():
    for name in producer_names:
        # Constructing dataset
        print(name)
        producer = get_real_producer_data(producers_df, name)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(producer, injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.7, shuffle=False
        )
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        # Setting up estimator
        bgr = MBBaggingRegressor(
            base_estimator=CRMP(), bootstrap=True, n_jobs=-1, random_state=0
        )
        param_grid = {
            'n_estimators': [5, 10],
            'block_size': [5, 10]
        }
        gcv = GridSearchCV(bgr, param_grid=param_grid, scoring=scorer)

        # Fitting and predicting with estimator
        gcv.fit(X_train, y_train)
        y_hat = gcv.predict(X_test)

        # Finding r2 and mse of the prediction
        r2, mse = fit_statistics(y_hat, y_test)
        print(r2)
        print(mse)
        print(gcv.best_params_)
        print()
        print()


train_bagging_regressor_with_crmp()
