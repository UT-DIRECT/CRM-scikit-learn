import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split

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
from src.models.crmp import CRMP
from src.models.crmpbhp import CrmpBHP
from src.models.MBBaggingRegressor import MBBaggingRegressor
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
    for name in producer_names:
        # Constructing dataset
        print(name)
        producer = get_real_producer_data(producers_df, name)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(producer, injectors, bhp=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        # Setting up estimator
        bgr = MBBaggingRegressor(
            base_estimator=CrmpBHP(), bootstrap=True, n_jobs=-1, random_state=0
        )
        param_grid = {
            'n_estimators': [10],
            'block_size': [5]
        }
        gcv = GridSearchCV(bgr, param_grid=param_grid, scoring=scorer)

        # Fitting and predicting with estimator
        gcv.fit(X_train, y_train)
        y_hat = gcv.predict(X_test)

        # Finding r2 and mse of the prediction
        r2, mse = fit_statistics(y_hat, y_test, shutin=True)
        # print(r2)
        # print(mse)
        # print(gcv.best_params_)
        # print(gcv.best_estimator_)
        best_estimator = gcv.best_estimator_
        best_estimator.fit(X_train, y_train)
        y_hats = []
        l = len(y_test)
        for e in best_estimator.estimators_:
            y_hat_i = []
            for i in range(30):
                y_hat_i.append(e.predict(X_test[i]))
                X_test[i + 1][0] = y_hat_i[i]
            y_hats.append(y_hat_i)
        y_hats_by_time = np.asarray(y_hats).T
        p10s = []
        averages = []
        p90s = []
        for y_hats_i in y_hats_by_time:
            p10 = np.percentile(y_hats_i, 10)
            average = np.average(y_hats_i)
            p90 = np.percentile(y_hats_i, 90)
            p10s.append(p10)
            averages.append(average)
            p90s.append(p90)
        print(p10s[0])
        print(averages[0])
        print(p90s[0])
        plt.plot(y_test[:30], color='k', label='True Value')
        plt.plot(averages, color='b', label='Average')
        plt.plot(p10s, color='r', alpha=0.5, label='P10 & P90')
        plt.plot(p90s, color='r', alpha=0.5)
        for hat in y_hats:
            plt.plot(hat, color='k', alpha=0.1)
        plot_helper(
            FIG_DIR,
            title='{}: 30 Day Prediction with BHP Data'.format(name),
            xlabel='Days',
            ylabel='Production Rate [bbls/day]',
            legend=True,
            save=True
        )
        raise
        print()
        print()


train_bagging_regressor_with_crmp()
