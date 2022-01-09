import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split

from crmp import CRMP, CrmpBHP, MBBaggingRegressor

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import scorer
from src.helpers.features import (
    get_real_producer_data, impute_training_data, production_rate_dataset,
    producer_rows_from_df, construct_real_production_rate_dataset
)
from src.helpers.figures import plot_helper
from src.helpers.models import model_namer
from src.simulations import injector_names, producer_names


FIG_DIR = INPUTS['real_data']['figures_dir']
injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])


def train_bagging_regressor_with_crmp():
    # producer_names = ['PA01', 'PA02', 'PA03', 'PA09', 'PA10', 'PA12']
    train_sizes = [0.33, 0.735, 0.49, 0.56, 0.80, 0.45, 0.54]
    for i in range(len(producer_names)):
        # Constructing dataset
        name = producer_names[i]
        print(name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(
            producer, injectors
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_sizes[i], shuffle=False
        )
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        # Setting up estimator
        bgr = MBBaggingRegressor(
            base_estimator=CrmpBHP(), n_estimators=100, bootstrap=True,
            random_state=0
        )
        param_grid = {
            'block_size': [7, 14, 21, 28, 90]
        }
        gcv = GridSearchCV(bgr, param_grid=param_grid, scoring=scorer)

        # Fitting the estimator
        gcv.fit(X_train, y_train)

        print(gcv.best_params_)
        print(gcv.best_estimator_)
        print()
        print()


train_bagging_regressor_with_crmp()
