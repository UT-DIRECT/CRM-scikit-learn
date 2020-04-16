import matplotlib.pyplot as plt
import numpy as np
import yaml

from lmfit import Model, Parameters
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, LassoCV,
        LinearRegression)
from sklearn.utils.testing import ignore_warnings

from ..data.process_dataset import INPUTS
from ..helpers.analysis import fit_statistics
from ..helpers.cross_validation import (forward_walk_and_ML,
        forward_walk_splitter)
from ..helpers.figures import (bar_plot_formater, bar_plot_helper, fig_saver,
        plot_helper)


class CRM():


    def __init__(self):
        data_file = INPUTS['files']['data']
        self.read_data(data_file)

        self.producers = np.array([self.q_1, self.q_2, self.q_3, self.q_4])
        self.producer_names = [
            'Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'
        ]
        self.net_productions = np.array([
            self.N_1, self.N_2, self.N_3, self.N_4
        ])

        self.initialize_crm_functions()
        self.step_sizes = np.linspace(2, 12, num=11).astype(int)
        self.N_predictions_output_file = INPUTS['files']['N_predictions']


    def initialize_crm_functions(self):
        self.q2 = lambda X, f1, f2, tau: X[0] * np.exp(-1 / tau) + (1 - np.exp(-1 / tau)) * (X[1] * f1 + X[2] * f2)
        self.N2 = lambda X: X[0] + X[1]
        self.p0 = 0.5, 0.5, 5


    def read_data(self, data_file):
        data = np.loadtxt(data_file, delimiter=',', skiprows=1).T
        self.Time = data[0]
        self.Random_inj1 = data[1]
        self.Random_inj2 = data[2]
        self.Fixed_inj1 = data[3]
        self.Net_Fixed_inj1 = data[4]
        self.Fixed_inj2 = data[5]
        self.Net_Fixed_inj2 = data[6]
        self.q_1 = data[7]
        self.N_1 = data[8]
        self.q_2 = data[9]
        self.N_2 = data[10]
        self.q_3 = data[11]
        self.N_3 = data[12]
        self.q_4 = data[13]
        self.N_4 = data[14]


    def production_rate_features(self, producer):
        size = producer[:-1].size
        return np.array([
            producer[:size], self.Fixed_inj1[:size], self.Fixed_inj2[:size]
        ])


    def net_production_features(self, net_production, q2):
        size = min(net_production.size, q2.size) - 1
        return np.array([net_production[:size], q2[:size]])


    def target_vector(self, production):
        return production[1:]


    def production_rate_dataset(self, producer):
        return [
            self.production_rate_features(producer),
            self.target_vector(producer)
        ]


    def net_production_dataset(self, net_production, q2):
        return [
            self.net_production_features(net_production, q2).T,
            self.target_vector(net_production)
        ]


    def fit_producer(self, producer):
        X, y = self.production_rate_dataset(producer)
        return self.fit_production_rate(X, y)


    def fit_production_rate(self, X, y):
        model = Model(self.q2, independent_vars=['X'])
        params = Parameters()

        params.add('f1', value=0.5, min=0, max=1)
        params.add('f2', expr='1-f1')
        params.add('tau', value=5, min=1, max=30)
        results = model.fit(y, X=X, params=params)

        pars = []
        pars.append(results.values['f1'])
        pars.append(results.values['f2'])
        pars.append(results.values['tau'])
        return pars


    def get_fitted_production_rate(self, producer):
        X = self.production_rate_features(producer)
        [f1, f2, tau] = self.fit_producer(producer)
        q2 = self.q2(X, f1, f2, tau)
        return q2


    def fit_producers(self):
        t = self.Time[1:]
        producers = self.producers
        for i in range(len(producers)):
            producer = producers[i]
            y = self.target_vector(producer)
            q2 = self.get_fitted_production_rate(producer)
            r2 = fit_statistics(q2, y)[0]
            self.data_and_crm_fitting_plotter(t, y, q2, r2)
            plot_helper(
                title='Producer {}'.format(i + 1),
                xlabel='Time',
                ylabel='Production Rate',
                legend=['CRM', 'Data'],
                save=True
            )


    def data_and_crm_fitting_plotter(self, t, y, y_hat, r2):
        plt.figure()
        plt.scatter(t, y)
        plt.plot(t, y_hat, 'r')
        plt.text(80, y[0], 'R-squared = %0.8f' % r2)


    def fit_net_productions(self):
        t = self.Time[1:]
        producers = self.producers
        net_productions = self.net_productions
        for i in range(len(producers)):
            producer = producers[i]
            net_production = net_productions[i]
            y = self.target_vector(net_production)
            net_production_fit = self.fit_net_production(
                producer, net_production
            )
            r2 = fit_statistics(net_production_fit, y)[0]
            self.data_and_crm_fitting_plotter(t, y, net_production_fit, r2)
            plot_helper(
                title='Producer {}'.format(i + 1),
                xlabel='Time',
                ylabel='Net Production',
                legend=['CRM', 'Data'],
                save=True
            )


    def fit_net_production(self, producer, net_production):
        X2 = self.net_production_features(net_production, producer)
        net_production_fit = self.N2(X2)
        return net_production_fit


    def crm_predict_net_production(self, producer, step_size):
        net_production = self.net_productions[producer]
        producer = self.producers[producer]
        X, y = self.production_rate_dataset(producer)
        X = X.T

        train_test_splits = forward_walk_splitter(X, y, step_size)
        test_split = train_test_splits[1]
        train_test_seperation_idx = train_test_splits[2]
        length = len(test_split)

        X_train = X[:train_test_seperation_idx].T
        y_train = y[:train_test_seperation_idx]
        [f1, f2, tau] = self.fit_production_rate(X_train, y_train)

        y2 = self.target_vector(net_production)
        r2_sum, mse_sum = 0, 0

        for train, test in test_split:
            y_train, q2_hat = self.production_rate_step(X, y, train, test)
            q2 = np.concatenate((y_train, q2_hat))

            y2_test, y2_hat = self.net_production_step(
                net_production, q2, y2, train, test
            )

            r2_i, mse_i = fit_statistics(y2_hat, y2_test)
            r2_sum += r2_i
            mse_sum += mse_i

        r2 = r2_sum / length
        mse = mse_sum / length
        return (r2, mse)


    def production_rate_step(self, X, y, train, test):
        X_train, X_test = X[train].T, X[test].T
        y_train = y[train]
        [f1, f2, tau] = self.fit_production_rate(X_train, y_train)
        q2_hat = self.q2(X_test, f1, f1, tau)
        return (y_train, q2_hat)

    
    def net_production_step(self, net_production, q2, y2, train, test):
        X2 = np.array([net_production[:len(q2)], q2]).T
        X2_test = X2[test]
        y2_test = y2[test]
        y2_hat = self.N2(X2_test.T)
        return (y2_test, y2_hat)


    @ignore_warnings(category=ConvergenceWarning)
    def net_production_predictions(self):
        output_header = "producer step_size crm_r2 cse_mse linear_regression_r2 linear_regression_mse bayesian_ridge_r2 bayesian_ridge_mse lasso_r2 lasso_mse elastic_r2 elastic_mse"
        producers = self.producers
        with open(self.N_predictions_output_file, 'w') as f:
            f.write('{}\n'.format(output_header))
            for i in range(len(producers)):
                models = [
                    BayesianRidge(), ElasticNetCV, LassoCV, LinearRegression()
                ]
                for step_size in self.step_sizes:
                    CRM_r2, CRM_mse = self.crm_predict_net_production(
                        i, step_size
                    )
                    models_performance_parameters = []
                    for model in models:
                        r2, mse = self.predict_ML_net_production(
                            i, step_size, model
                        )
                        models_performance_parameters.append(r2)
                        models_performance_parameters.append(mse)
                    f.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                            i + 1, int(step_size), CRM_r2, CRM_mse,
                            *models_performance_parameters
                        )
                    )


    def net_production_predictions_plot(self):
        x_labels = [int(step_size) for step_size in self.step_sizes]
        prediction_results = np.genfromtxt(
            self.N_predictions_output_file, skip_header=1
        )
        x = np.arange(len(x_labels))
        width = 0.15
        bar_labels = [
            'CRM, mse', 'Linear Regression, mse', 'Bayesian Ridge, mse',
            'Lasso, mse', 'Elastic, mse'
        ]
        for i in range(4):
            producer = i + 1
            producer_rows = np.where(prediction_results[:,0] == producer)
            producer_results = prediction_results[producer_rows]
            crm_mse = producer_results[:, 3]
            linear_regression_mse = producer_results[:, 5]
            bayesian_ridge_mse = producer_results[:, 7]
            lasso_mse = producer_results[:, 9]
            elastic_mse = producer_results[:, 11]
            heights = [
                crm_mse, linear_regression_mse, bayesian_ridge_mse, lasso_mse,
                elastic_mse
            ]

            title = 'Producer {}'.format(producer)
            xlabel = 'Step Size'
            ylabel = 'Mean Squared Error'
            bar_plot_helper(width, x, x_labels, bar_labels, heights)
            bar_plot_formater(x, x_labels, title, xlabel, ylabel)


    def net_production_good_estimators_plot(self):
        x_labels = [int(step_size) for step_size in self.step_sizes]
        prediction_results = np.genfromtxt(
            self.N_predictions_output_file, skip_header=1
        )
        x = np.arange(len(x_labels))
        width = 0.3
        bar_labels = ['CRM, mse', 'Linear Regression, mse', 'Elastic, mse']
        for i in range(4):
            producer = i + 1
            producer_rows = np.where(prediction_results[:,0] == producer)
            producer_results = prediction_results[producer_rows]
            crm_mse = producer_results[:, 3]
            linear_regression_mse = producer_results[:, 5]
            elastic_mse = producer_results[:, 11]
            heights = [crm_mse, linear_regression_mse, elastic_mse]

            title = 'Good Estimator MSEs Producer {}'.format(producer)
            xlabel = 'Step Size'
            ylabel = 'Mean Squared Error'
            bar_plot_helper(width, x, x_labels, bar_labels, heights)
            bar_plot_formater(x, x_labels, title, xlabel, ylabel)


    def predict_ML_net_production(self, producer, step_size, model):
        net_production = self.net_productions[producer]
        producer = self.producers[producer]
        X = np.array([
            producer[:-1], net_production[:-1],
            self.Net_Fixed_inj1[:-1], self.Net_Fixed_inj2[:-1]
        ]).T
        y = self.target_vector(net_production)
        return forward_walk_and_ML(X, y, step_size, model)


    def plot_producers_vs_time(self):
        plt.figure()
        plt.plot(self.Time, self.producers.T)
        plot_helper(
            title='Production Rate vs Time',
            xlabel='Time',
            ylabel='Production Rate',
            legend=self.producer_names,
            save=True
        )


    def plot_net_production_vs_time(self):
        plt.figure()
        plt.plot(self.Time, self.net_productions.T)
        plot_helper(
            title='Total Production vs Time',
            xlabel='Time',
            ylabel='Net Production',
            legend=self.producer_names,
            save=True
        )


    def plot_producers_vs_injector(self):
        injectors = [self.Fixed_inj1, self.Fixed_inj2]
        for i in range(len(injectors)):
            plt.figure()
            for producer in self.producers:
                plt.scatter(injectors[i], producer)
            plot_helper(
                title='Injector {}'.format(i + 1),
                xlabel='Injection Rate',
                ylabel='Production Rate',
                legend=self.producer_names,
                save=True
            )


model = CRM()
model.fit_producers()
model.fit_net_productions()
model.plot_producers_vs_time()
model.plot_net_production_vs_time()
model.plot_producers_vs_injector()
model.net_production_predictions()
model.net_production_predictions_plot()
model.net_production_good_estimators_plot()
