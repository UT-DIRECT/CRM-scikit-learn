import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.figures import plot_helper
from src.helpers.models import model_namer, test_model
from src.models.crmp import CRMP
from src.simulations import number_of_producers, param_grid


fit_ouput_file = INPUTS['crmp']['crmp']['fit']['sensitivity_analysis']
predict_output_file = INPUTS['crmp']['crmp']['predict']['sensitivity_analysis']
objective_function_file  = INPUTS['crmp']['crmp']['predict']['objective_function']
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
objective_function_data = {
    'Producer': [], 'tau': [], 'f1': [], 'f2': [], 'r2': [], 'MSE': []
}

def convergence_sensitivity_analysis():
    t = time[1:]
    for i in range(number_of_producers):
        X, y = production_rate_dataset(producers[i], *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, shuffle=False
        )
        train_length = len(y_train)
        test_length = len(y_test)
        train_time = t[:train_length]
        test_time = t[train_length:]
        fit = plt.plot(train_time, y_train, c='r', label='Fit')
        predict = plt.plot(test_time, y_test, c='g', label='Predict')
        for p0 in param_grid['p0']:
            crmp = CRMP(p0=p0)
            crmp = crmp.fit(X_train, y_train)

            # Fitting
            y_hat = crmp.predict(X_train)
            plt.plot(train_time, y_hat, alpha=0.2, c='r')
            r2, mse = fit_statistics(y_hat, y_train)
            fit_data['Producer'].append(i + 1)
            fit_data['Model'].append(model_namer(crmp))
            fit_data['tau_initial'].append(p0[0])
            fit_data['tau_final'].append(crmp.tau_)
            fit_data['f1_initial'].append(p0[1])
            fit_data['f1_final'].append(crmp.gains_[0])
            fit_data['f2_initial'].append(p0[2])
            fit_data['f2_final'].append(crmp.gains_[1])
            fit_data['r2'].append(r2)
            fit_data['MSE'].append(mse)

            # Prediction
            y_hat = crmp.predict(X_test)
            plt.plot(test_time, y_hat, alpha=0.2, c='g')
            r2, mse = fit_statistics(y_hat, y_test)
            predict_data['Producer'].append(i + 1)
            predict_data['Model'].append(model_namer(crmp))
            predict_data['tau_initial'].append(p0[0])
            predict_data['tau_final'].append(crmp.tau_)
            predict_data['f1_initial'].append(p0[1])
            predict_data['f1_final'].append(crmp.gains_[0])
            predict_data['f2_initial'].append(p0[2])
            predict_data['f2_final'].append(crmp.gains_[1])
            predict_data['r2'].append(r2)
            predict_data['MSE'].append(mse)

        plt.vlines(76, 0, 1000, linewidth=3)
        plt.title(producer_names[i])
        plt.xlabel('Time')
        plt.ylabel('Production Rate')
        plt.legend()
        plt.show()

    # Fitting
    fit_df = pd.DataFrame(fit_data)
    fit_df.to_csv(fit_ouput_file)

    # Prediction
    predict_df = pd.DataFrame(predict_data)
    predict_df.to_csv(predict_output_file)


def objective_function():
    X, y = production_rate_dataset(producers[0], *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    crmp = CRMP().fit(X_train, y_train)
    for i in range(number_of_producers):
        X, y = production_rate_dataset(producers[i], *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, shuffle=False
        )
        for p0 in param_grid['p0']:
            crmp.tau_ = p0[0]
            crmp.gains_ = p0[1:]
            y_hat = crmp.predict(X_test)
            r2, mse = fit_statistics(y_hat, y_test)
            objective_function_data['Producer'].append(i + 1)
            objective_function_data['tau'].append(p0[0])
            objective_function_data['f1'].append(p0[1])
            objective_function_data['f2'].append(p0[2])
            objective_function_data['r2'].append(r2)
            objective_function_data['MSE'].append(mse)

    objective_function_df = pd.DataFrame(objective_function_data)
    objective_function_df.to_csv(objective_function_file)


def minimum_train_size():
    data_sizes = np.linspace(1, 75, 75).astype(int)
    for data_size in data_sizes:
        X, y = production_rate_dataset(producers[3], *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=data_size, shuffle=False
        )
        crmp = CRMP(p0=[5, 0.5, 0.5])
        crmp = crmp.fit(X_train, y_train)
        y_hat = crmp.predict(X_train)
        r2, mse = fit_statistics(y_hat, y_train)
        if mse < 1e-03:
            print(data_size)
            print(mse)
            return


# convergence_sensitivity_analysis()
# objective_function()
minimum_train_size()
