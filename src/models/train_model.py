import matplotlib.pyplot as plt
import numpy as np

from lmfit import Parameters, Model
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import *
from sklearn.metrics  import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR

from ..helpers.figures import plot_helper, fig_filename, FIG_DIR

filename = "./data/interim/CRMP_Corrected_July16_2018.csv"

class CRM():

    def __init__(self, filename):
        [
            self.Time, self.Random_inj1, self.Random_inj2, self.Fixed_inj1,
            self.Net_Fixed_inj1, self.Fixed_inj2, self.Net_Fixed_inj2,
            self.Prod1, self.Net_Prod1, self.Prod2, self.Net_Prod2,
            self.Prod3, self.Net_Prod3, self.Prod4, self.Net_Prod4
        ] = np.loadtxt(filename, delimiter=',', skiprows=1).T
        self.producers = np.array(
            [self.Prod1, self.Prod2, self.Prod3, self.Prod4]
        )
        self.producer_names = [
            'Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'
        ]
        self.net_productions = np.array(
            [self.Net_Prod1, self.Net_Prod2, self.Net_Prod3, self.Net_Prod4]
        )
        self.q2 = lambda X, f1, f2, tau: X[0] * np.exp(-1 / tau) + (1 - np.exp(-1 / tau)) * (X[1] * f1 + X[2] * f2)
        self.N2 = lambda X: X[0] + X[1]
        self.p0 = .5, .5, 5
        self.step_sizes = np.linspace(2, 12, num=11)
        self.net_production_forward_walk_predictions_output_file = './data/interim/net_production_forward_walk_predictions.txt'

    def production_rate_features(self, producer):
        return np.array([
            producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
        ])

    def net_production_features(self, net_production, q2):
        return np.array([net_production[:-1], q2])

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
        t, producers = self.Time[1:], self.producers
        for i in range(len(producers)):
            producer = producers[i]
            y = self.target_vector(producer)
            q2 = self.get_fitted_production_rate(producer)
            r2 = self.fit_statistics(q2, y)[0]
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

    def fit_statistics(self, y_hat, y):
        r2 = r2_score(y_hat, y)
        mse = mean_squared_error(y_hat, y)
        return [r2, mse]

    def fit_net_productions(self):
        t = self.Time[1:]
        producers = self.producers
        net_productions = self.net_productions
        for i in range(len(producers)):
            producer = producers[i]
            net_production = net_productions[i]
            y = self.target_vector(net_production)
            net_production_fit = self.fit_net_production(
                producer,
                net_production
            )
            r2 = self.fit_statistics(net_production_fit, y)[0]
            self.data_and_crm_fitting_plotter(t, y, net_production_fit, r2)
            plot_helper(
                title='Producer {}'.format(i + 1),
                xlabel='Time',
                ylabel='Net Production',
                legend=['CRM', 'Data'],
                save=True
            )

    def fit_net_production(self, producer, net_production):
        q2 = self.get_fitted_production_rate(producer)
        X2 = self.net_production_features(net_production, q2)
        net_production_fit = self.N2(X2)
        return net_production_fit

    def crm_predict_net_production(self, producer, step_size):
        net_production = self.net_productions[producer]
        producer = self.producers[producer]
        q2 = self.get_fitted_production_rate(producer)
        X2, y = self.net_production_dataset(net_production, q2)
        [tscv, n_splits] = self.time_series_cross_validator(X2, step_size)
        r2_sum = 0
        mse_sum = 0
        for train_index, test_index in tscv.split(X2):
            x_train, x_test = (X2[train_index]).T, (X2[test_index]).T
            y_train, y_test = y[train_index], y[test_index]
            y_predict = self.N2(x_test)
            r2_i, mse_i = self.fit_statistics(y_predict, y_test)
            r2_sum += r2_i
            mse_sum += mse_i
        r2 = r2_sum / n_splits
        mse = mse_sum / n_splits
        return (r2, mse)

    def net_production_forward_walk_predictions(self):
        output_header = "producer step_size crm_r2 cse_mse linear_regression_r2 linear_regression_mse bayesian_ridge_r2 bayesian_ridge_mse lasso_r2 lasso_mse elastic_r2 elastic_mse"
        producers = self.producers
        with open(self.net_production_forward_walk_predictions_output_file, 'w') as f:
            f.write('{}\n'.format(output_header))
            for i in range(len(producers)):
                models = [
                    LinearRegression(), BayesianRidge(),
                    Lasso(max_iter=100000), ElasticNet(max_iter=100000)
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
                    f.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(i + 1, int(step_size), CRM_r2, CRM_mse, *models_performance_parameters))

    def mse_of_net_production_vs_time(self):
        producers = self.producers
        net_productions = self.net_productions
        for i in range(len(producers)):
            producer = producers[i]
            net_production = net_productions[i]
            X = self.production_rate_features(producer)
            [f1, f2, tau] = self.fit_producer(producer)
            q2 = self.q2(X, f1, f2, tau)
            X2 = self.net_production_features(net_production, q2).T
            y2 = net_production[1:]
            [tscv, n_splits] = self.time_series_cross_validator(X2, 12)
            y = []
            for train_index, test_index in tscv.split(X2):
                x_train, x_test = (X2[train_index]).T, (X2[test_index]).T
                y_train, y_test = y2[train_index], y2[test_index]
                y_predict = self.N2(x_test)
                y.append(r2_score(y_predict, y_test))
            size = len(y)
            x = np.linspace(0, size, num=size)
            plt.figure()
            plt.plot(x, y, 'r')
            plot_helper(
                title='Producer {}'.format(i + 1),
                xlabel='Time',
                ylabel='MSE',
                save=False
            )

    def net_production_forward_walk_predictions_plot(self):
        # TODO: Refactor
        labels = [int(step_size) for step_size in self.step_sizes]
        prediction_results = np.genfromtxt(self.net_production_forward_walk_predictions_output_file, skip_header=1)
        x = np.arange(len(labels))
        width = 0.15
        for i in range(4):
            fig, ax = plt.subplots()
            producer = i + 1
            producer_rows = np.where(prediction_results[:,0] == producer)
            producer_results = prediction_results[producer_rows]
            CRM_mse = producer_results[:, 3]
            Linear_Regression_mse = producer_results[:, 5]
            Bayesian_Ridge_mse = producer_results[:, 7]
            Lasso_mse = producer_results[:, 9]
            Elastic_mse = producer_results[:, 11]
            rects1 = ax.bar(x - 2 * width, CRM_mse, width, label='CRM, mse')
            rects2 = ax.bar(x - width, Linear_Regression_mse, width, label='Linear Regression, mse')
            rects3 = ax.bar(x, Bayesian_Ridge_mse, width, label='Bayesian Ridge, mse')
            rects4 = ax.bar(x + width, Lasso_mse, width, label='Lasso, mse')
            rects5 = ax.bar(x + 2 * width, Elastic_mse, width, label='Elastic, mse')
            xlabel = 'Step Size'
            ylabel = 'Mean Squared Error'
            title = 'Producer {}'.format(producer)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            fig_file = fig_filename(title, xlabel, ylabel)
            plt.savefig('{}{}'.format(FIG_DIR, fig_file))

    def time_series_cross_validator(self, X, step_size):
        n_splits = (int(len(X) / step_size) - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return [tscv, n_splits]

    def forward_walk_and_ML(self, X, y, model, step_size):
        [tscv, n_splits] = self.time_series_cross_validator(X, step_size)
        r2_sum, mse_sum= 0, 0
        for train_index, test_index in tscv.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_predict = self.train_and_test_model(model, x_train, y_train, x_test)
            r2_i, mse_i = self.fit_statistics(y_predict, y_test)
            r2_sum += r2_i
            mse_sum += mse_i
        r2 = r2_sum / n_splits
        mse = mse_sum / n_splits
        return (r2, mse)

    def train_and_test_model(self, model, x_train, y_train, x_test):
        model.fit(x_train, y_train)
        return model.predict(x_test)

    def predict_ML_net_production(self, producer, step_size, model):
        net_production = self.net_productions[producer]
        producer = self.producers[producer]
        X = np.array([
            producer[:-1], net_production[:-1], self.Net_Fixed_inj1[:-1], self.Net_Fixed_inj2[:-1]
        ]).T
        y = self.target_vector(net_production)
        # models = [LinearRegression(), Lasso(alpha=0, max_iter=100, tol=0.0001),
        #         Ridge(), ElasticNet(),  Lars(),
        #         LassoLars(), OrthogonalMatchingPursuit(), BayesianRidge(),
        #         ARDRegression(), SGDRegressor(), PassiveAggressiveRegressor(),
        #         KernelRidge(), SVR(), NuSVR(), LinearSVR(),
        #         KNeighborsRegressor(n_neighbors=3),
        #         RadiusNeighborsRegressor(radius=10000),
        #         GaussianProcessRegressor(), MLPRegressor(hidden_layer_sizes=(60,))]
        return self.forward_walk_and_ML(X, y, model, step_size)

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

model = CRM(filename)
model.fit_producers()
model.fit_net_productions()
model.plot_producers_vs_time()
model.plot_net_production_vs_time()
model.plot_producers_vs_injector()
model.net_production_forward_walk_predictions()
model.net_production_forward_walk_predictions_plot()
model.mse_of_net_production_vs_time()
