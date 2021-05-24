from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_clair import (
    injectors, producers, producer_names, producer_starting_indicies, time
)
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset, producer_rows_from_df
from src.helpers.models import model_namer
from src.models.crmp import CRMP


fit_ouput_file = INPUTS['real_data']['fit']['sensitivity_analysis']
predict_output_file = INPUTS['real_data']['predict']['sensitivity_analysis']
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


number_of_gains = 6
number_of_time_constants = 11
gains = np.linspace(0, 1, number_of_gains)
tau = np.linspace(0, 100, number_of_time_constants)
p0s = []
for i in tau:
    if i == 0:
        i = 1e-06
    for j in gains:
        for k in gains:
            for l in gains:
                for m in gains:
                    if (j + k + l + m) != 1:
                        continue
                    else:
                        p0s.append((i, j, k, l, m))

p0s = set(p0s)
p0s = [[100.0, 0.2, 0.6000000000000001, 0.0, 0.2], [80.0, 0.0, 0.0, 0.2, 0.8], [40.0, 0.4, 0.0, 0.2, 0.4], [40.0, 0.4, 0.2, 0.4, 0.0], [100.0, 0.2, 0.0, 0.4, 0.4], [90.0, 0.8, 0.0, 0.0, 0.2], [30.0, 0.4, 0.0, 0.4, 0.2], [50.0, 0.4, 0.0, 0.2, 0.4], [30.0, 0.0, 0.0, 1.0, 0.0], [70.0, 0.2, 0.4, 0.0, 0.4], [1e-06, 0.2, 0.0, 0.8, 0.0], [30.0, 0.0, 0.8, 0.0, 0.2], [90.0, 0.0, 0.2, 0.8, 0.0], [90.0, 0.2, 0.2, 0.6000000000000001, 0.0], [10.0, 0.2, 0.6000000000000001, 0.2, 0.0], [30.0, 0.2, 0.6000000000000001, 0.2, 0.0], [50.0, 0.0, 0.6000000000000001, 0.0, 0.4], [40.0, 0.2, 0.0, 0.0, 0.8], [50.0, 0.0, 1.0, 0.0, 0.0], [90.0, 0.2, 0.0, 0.6000000000000001, 0.2]]




def convergence_sensitivity_analysis():
    for i in range(len(producer_names)):
        starting_index = producer_starting_indicies[i]
        producer = producers[i][starting_index:]
        injectors_tmp = [injector[starting_index:] for injector in injectors]
        X, y = production_rate_dataset(producer, *injectors_tmp)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, shuffle=False
        )
        for p0 in p0s:
            crmp = CRMP(p0=deepcopy(p0))
            crmp = crmp.fit(X_train, y_train)

            # Fitting
            y_hat = crmp.predict(X_train)
            r2, mse = fit_statistics(y_hat, y_train)
            fit_data['Producer'].append(i + 1)
            fit_data['Model'].append(model_namer(crmp))
            fit_data['tau_initial'].append(p0[0])
            fit_data['tau_final'].append(crmp.tau_)
            fit_data['f1_initial'].append(p0[1])
            fit_data['f1_final'].append(crmp.gains_[0])
            fit_data['f2_initial'].append(p0[2])
            fit_data['f2_final'].append(crmp.gains_[1])
            fit_data['f3_initial'].append(p0[3])
            fit_data['f3_final'].append(crmp.gains_[2])
            fit_data['f4_initial'].append(p0[4])
            fit_data['f4_final'].append(crmp.gains_[3])
            fit_data['r2'].append(r2)
            fit_data['MSE'].append(mse)

            # Prediction
            y_hat = crmp.predict(X_test)
            r2, mse = fit_statistics(y_hat, y_test)
            predict_data['Producer'].append(i + 1)
            predict_data['Model'].append(model_namer(crmp))
            predict_data['tau_initial'].append(p0[0])
            predict_data['tau_final'].append(crmp.tau_)
            predict_data['f1_initial'].append(p0[1])
            predict_data['f1_final'].append(crmp.gains_[0])
            predict_data['f2_initial'].append(p0[2])
            predict_data['f2_final'].append(crmp.gains_[1])
            predict_data['f3_initial'].append(p0[3])
            predict_data['f3_final'].append(crmp.gains_[2])
            predict_data['f4_initial'].append(p0[4])
            predict_data['f4_final'].append(crmp.gains_[3])
            predict_data['r2'].append(r2)
            predict_data['MSE'].append(mse)

    # Fitting
    fit_df = pd.DataFrame(fit_data)
    fit_df.to_csv(fit_ouput_file)

    # Prediction
    predict_df = pd.DataFrame(predict_data)
    predict_df.to_csv(predict_output_file)


def converged_parameter_statistics():
    df = pd.read_csv(predict_output_file)
    for i in range(len(producer_names)):
        producer_df = producer_rows_from_df(df, i+1)
        print(producer_names[i])
        print('tau: ', producer_df['tau_final'].mean())
        print('tau std: ', producer_df['tau_final'].std())
        print('f1: ', producer_df['f1_final'].mean())
        print('f1 std: ', producer_df['f1_final'].std())
        print('f2: ', producer_df['f2_final'].mean())
        print('f2 std: ', producer_df['f2_final'].std())
        print('f3: ', producer_df['f3_final'].mean())
        print('f3 std: ', producer_df['f3_final'].std())
        print('f4: ', producer_df['f4_final'].mean())
        print('f4 std: ', producer_df['f4_final'].std())
        print('MSE: ', producer_df['MSE'].mean())
        print('MSE std: ', producer_df['MSE'].std())
        print()
        print()
        print()
        print()
        print()
        print()


def constraint_analysis():
    df = pd.read_csv(predict_output_file)
    length = len(p0s)
    print(length)
    print()
    for i in range(len(producer_names)):
        print(producer_names[i])
        constraint_violation_counter = 0
        sum_of_gains = 0
        producer_df = producer_rows_from_df(df, i+1)
        for index, row in producer_df.iterrows():
            f1 = row['f1_final']
            f2 = row['f2_final']
            f3 = row['f3_final']
            f4 = row['f4_final']
            gains = f1 + f2 + f3 + f4
            sum_of_gains += gains
            if gains > 1.01:
                constraint_violation_counter += 1
        print('Constraint Violations: ', constraint_violation_counter)
        print('Average Gains Sum: ', (sum_of_gains / length))
        print()


convergence_sensitivity_analysis()
converged_parameter_statistics()
constraint_analysis()
