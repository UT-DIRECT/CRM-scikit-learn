from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import INPUTS
from src.helpers.analysis import fit_statistics

from src.helpers.features import (
    get_real_producer_data, production_rate_dataset, producer_rows_from_df,
    construct_real_production_rate_dataset
)
from src.helpers.models import model_namer
from src.models.crmp import CRMP
from src.models.crmpbhp import CrmpBHP
from src.simulations import injector_names, producer_names


injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])

fit_output_file = INPUTS['real_data']['fit']['bhp_sensitivity_analysis']
predict_output_file = INPUTS['real_data']['predict']['bhp_sensitivity_analysis']
fit_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'f3_initial': [], 'f3_final': [], 'f4_initial': [], 'f4_final': [],
    'r2': [], 'MSE': []
}
predict_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'f3_initial': [], 'f3_final': [], 'f4_initial': [], 'f4_final': [],
    'r2': [], 'MSE': []
}


p0s = [
    [100.0, 2.0, 0.2, 0.6000000000000001, 0.0, 0.2], [80.0, 2.0, 0.0, 0.0, 0.2, 0.8],
    [40.0, 2.0, 0.4, 0.0, 0.2, 0.4], [40.0, 2.0, 0.4, 0.2, 0.4, 0.0],
    [100.0, 2.0, 0.2, 0.0, 0.4, 0.4], [90.0, 2.0, 0.8, 0.0, 0.0, 0.2],
    [30.0, 2.0, 0.4, 0.0, 0.4, 0.2], [50.0, 2.0, 0.4, 0.0, 0.2, 0.4],
    [30.0, 2.0, 0.0, 0.0, 1.0, 0.0], [70.0, 2.0, 0.2, 0.4, 0.0, 0.4],
    [1e-06, 2.0, 0.2, 0.0, 0.8, 0.0], [30.0, 2.0, 0.0, 0.8, 0.0, 0.2],
    [90.0, 2.0, 0.0, 0.2, 0.8, 0.0], [90.0, 2.0, 0.2, 0.2, 0.6000000000000001, 0.0],
    [10.0, 2.0, 0.2, 0.6000000000000001, 0.2, 0.0],
    [30.0, 2.0, 0.2, 0.6000000000000001, 0.2, 0.0],
    [50.0, 2.0, 0.0, 0.6000000000000001, 0.0, 0.4], [40.0, 2.0, 0.2, 0.0, 0.0, 0.8],
    [50.0, 2.0, 0.0, 1.0, 0.0, 0.0], [90.0, 2.0, 0.2, 0.0, 0.6000000000000001, 0.2]
]


def evaluate_crmp_bhp_model():
    iteration = 0
    for name in producer_names:
        print('Producer Name: ', name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        X, y = construct_real_production_rate_dataset(
            producer[['Date', name]], injectors, producer['delta_p']
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        for p0 in p0s:
            iteration += 1
            print('Iteration: {}'.format(iteration))
            crmpbhp = CrmpBHP(p0=deepcopy(p0))
            crmpbhp = crmpbhp.fit(X_train, y_train)

            # Fitting
            y_hat = crmpbhp.predict(X_train)
            r2, mse = fit_statistics(y_hat, y_train, shutin=True)
            fit_data['Producer'].append(name)
            fit_data['Model'].append(model_namer(crmpbhp))
            fit_data['tau_initial'].append(p0[0])
            fit_data['tau_final'].append(crmpbhp.tau_)
            fit_data['f1_initial'].append(p0[1])
            fit_data['f1_final'].append(crmpbhp.gains_[0])
            fit_data['f2_initial'].append(p0[2])
            fit_data['f2_final'].append(crmpbhp.gains_[1])
            fit_data['f3_initial'].append(p0[3])
            fit_data['f3_final'].append(crmpbhp.gains_[2])
            fit_data['f4_initial'].append(p0[4])
            fit_data['f4_final'].append(crmpbhp.gains_[3])
            fit_data['r2'].append(r2)
            fit_data['MSE'].append(mse)

            # Prediction
            y_hat = crmpbhp.predict(X_test)
            r2, mse = fit_statistics(y_hat, y_test, shutin=True)
            predict_data['Producer'].append(name)
            predict_data['Model'].append(model_namer(crmpbhp))
            predict_data['tau_initial'].append(p0[0])
            predict_data['tau_final'].append(crmpbhp.tau_)
            predict_data['f1_initial'].append(p0[1])
            predict_data['f1_final'].append(crmpbhp.gains_[0])
            predict_data['f2_initial'].append(p0[2])
            predict_data['f2_final'].append(crmpbhp.gains_[1])
            predict_data['f3_initial'].append(p0[3])
            predict_data['f3_final'].append(crmpbhp.gains_[2])
            predict_data['f4_initial'].append(p0[4])
            predict_data['f4_final'].append(crmpbhp.gains_[3])
            predict_data['r2'].append(r2)
            predict_data['MSE'].append(mse)

    # Fitting
    fit_df = pd.DataFrame(fit_data)
    fit_df.to_csv(fit_output_file)

    # Prediction
    predict_df = pd.DataFrame(predict_data)
    predict_df.to_csv(predict_output_file)


def get_converged_parameter_statistics():
    df = pd.read_csv(predict_output_file)
    for name in producer_names:
        producer_df = df.loc[df['Producer'] == name]
        print(name)
        # print('tau: ', producer_df['tau_final'].mean())
        # print('tau std: ', producer_df['tau_final'].std())
        # print('f1: ', producer_df['f1_final'].mean())
        # print('f1 std: ', producer_df['f1_final'].std())
        # print('f2: ', producer_df['f2_final'].mean())
        # print('f2 std: ', producer_df['f2_final'].std())
        # print('f3: ', producer_df['f3_final'].mean())
        # print('f3 std: ', producer_df['f3_final'].std())
        # print('f4: ', producer_df['f4_final'].mean())
        # print('f4 std: ', producer_df['f4_final'].std())
        print('MSE: ', producer_df['MSE'].mean())
        print('MSE std: ', producer_df['MSE'].std())
        print('r2: ', producer_df['r2'].mean())
        print('r2 std: ', producer_df['r2'].std())
        print()
        print()
        print()


evaluate_crmp_bhp_model()
get_converged_parameter_statistics()
