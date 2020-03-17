import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import *
from sklearn.metrics  import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR

from ..helpers.figures import plot_helper, fig_filename

filename = "/Users/akhilpotla/ut/research/crm_validation/data/interim/CRMP_Corrected_July16_2018.csv"

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
        self.net_production_by_producer = np.array(
            [self.Net_Prod1, self.Net_Prod2, self.Net_Prod3, self.Net_Prod4]
        )
        self.q2 = lambda X, f1, f2, tau: X[0] * np.exp(-1 / tau) + (1 - np.exp(-1 / tau)) * (X[1] * f1 + X[2] * f2)
        self.N2 = lambda X: X[0] + X[1]
        self.p0 = .5, .5, 5

    def fit_producer(self, producer):
        X = np.array([
            producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
        ])
        y = producer[1:]
        return curve_fit(
            self.q2, X, y, p0=self.p0,
            bounds=(
                (0.01, 0.01, 1,),
                (0.8, 0.8, 30)
            )
        )[0]

    def fit_producers(self):
        t = self.Time[1:]
        producers = self.producers
        with open('/Users/akhilpotla/ut/research/crm_validation/data/interim/crm_producers_fit_results.txt', 'w') as f:
            for i in range(len(producers)):
                producer = producers[i]
                X = np.array([
                    producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
                ])
                y = producer[1:]
                [f1, f2, tau] = self.fit_producer(producer)
                q2 = self.q2(X, f1, f2, tau)
                r2 = r2_score(q2, y)
                mse = mean_squared_error(q2, y)
                f.write('==========================================================\n')
                f.write('Producer {}\n'.format(i + 1))
                f.write('r2 score: {}\n'.format(r2))
                f.write('MSE: {}\n'.format(mse))
                f.write('\n\n')
                plt.figure()
                plt.scatter(t, y)
                plt.plot(t, q2, 'r')
                plt.text(80, 0.5, 'R-squared = %0.8f' % r2)
                plot_helper(
                    title='Producer {}'.format(i + 1),
                    xlabel='Time',
                    ylabel='Production Rate',
                    legend=['CRM', 'Data'],
                    save=True
                )
                plt.close()

    def fit_net_production(self, N1, q2):
        return N1 + q2

    def fit_net_productions(self):
        t = self.Time[1:]
        producers = self.producers
        net_productions = self.net_production_by_producer
        for i in range(len(producers)):
            producer = producers[i]
            net_production = net_productions[i]
            [f1, f2, tau] = self.fit_producer(producer)
            X = np.array([
                producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
            ])
            q2 = self.q2(X, f1, f2, tau)
            X = [net_production[:-1], q2]
            y = net_production[1:]
            predicted_net_production = self.N2(X)
            r2 = r2_score(y, predicted_net_production)
            plt.figure()
            plt.scatter(t, y)
            plt.plot(t, predicted_net_production, 'r')
            plt.text(80, 0.5, 'R-squared = %0.8f' % r2)
            plot_helper(
                title='Producer {}'.format(i + 1),
                xlabel='Time',
                ylabel='Net Production',
                legend=['CRM', 'Data'],
                save=True
            )
            plt.close()

    def crm_predict_net_production(self):
        net_productions = self.net_production_by_producer
        with open('/Users/akhilpotla/ut/research/crm_validation/data/interim/crm_forward_walk_results.txt', 'w') as f:
            for i in range(len(self.producers)):
                producer = self.producers[i]
                net_production = net_productions[i]
                X = np.array([
                    producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
                ])
                [f1, f2, tau] = self.fit_producer(producer)
                q2 = self.q2(X, f1, f2, tau)
                X2 = np.array([net_production[:-1], q2]).T
                y2 = net_production[1:]
                n_splits = (int(len(X2) / 2) - 1)
                tscv = TimeSeriesSplit(n_splits=n_splits)
                r2_sum = 0
                mse_sum = 0
                for train_index, test_index in tscv.split(X2):
                    x_train, x_test = (X2[train_index]).T, (X2[test_index]).T
                    y_train, y_test = y2[train_index], y2[test_index]
                    y_predict = self.N2(x_test)
                    r2_sum += r2_score(y_predict, y_test)
                    mse_sum += mean_squared_error(y_predict, y_test)
                f.write('==========================================================\n')
                f.write('PRODUCER {}\n'.format(i + 1))
                f.write('Average r2: {}\n'.format(r2_sum / n_splits))
                f.write('Average MSE: {}\n'.format(mse_sum/ n_splits))
                f.write('\n\n')

    def forward_walk_and_ML(self, X, y, model):
        n_splits = (int(len(X) / 2) - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        r2_sum = 0
        mse_sum = 0
        for train_index, test_index in tscv.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            r2_sum += r2_score(y_predict, y_test)
            mse_sum += mean_squared_error(y_predict, y_test)
        return ((r2_sum / n_splits), (mse_sum / n_splits))

    def fit_ML_production_rate(self):
        producers = self.producers
        with open('/Users/akhilpotla/ut/research/crm_validation/data/interim/train_output_production_rate.txt', 'w') as f:
            for i in range(len(producers)):
                production = producers[i]
                X = np.array([
                    production[:-1], self.Net_Fixed_inj1[:-1], self.Net_Fixed_inj2[:-1]
                ]).T
                y = production[1:]
                n_splits = (int(len(X) / 2) - 1)
                tscv = TimeSeriesSplit(n_splits=n_splits)
                models = [LinearRegression(), BayesianRidge()]
                # models = [LinearRegression(), Lasso(alpha=0, max_iter=100, tol=0.0001),
                #         Ridge(), ElasticNet(),  Lars(),
                #         LassoLars(), OrthogonalMatchingPursuit(), BayesianRidge(),
                #         ARDRegression(), SGDRegressor(), PassiveAggressiveRegressor(),
                #         KernelRidge(), SVR(), NuSVR(), LinearSVR(),
                #         KNeighborsRegressor(n_neighbors=3),
                #         RadiusNeighborsRegressor(radius=10000),
                #         GaussianProcessRegressor(), MLPRegressor(hidden_layer_sizes=(60,))]
                f.write('==========================================================\n')
                f.write('Producer {}\n'.format(i + 1))
                for model in models:
                    r2, mse = self.forward_walk_and_ML(X, y, model)
                    f.write('model: {}\n'.format(type(model)))
                    f.write('Average r2: {}\n'.format(r2))
                    f.write('Average MSE: {}\n'.format(mse))
                    f.write('\n')
                f.write('\n')
                f.write('\n')
                f.write('\n')

    def predict_ML_net_production(self):
        producers = self.producers
        net_productions = self.net_production_by_producer
        with open('/Users/akhilpotla/ut/research/crm_validation/data/interim/train_output_net_production.txt', 'w') as f:
            for i in range(len(producers)):
                production = producers[i]
                net_production = self.net_production_by_producer[i]
                X = np.array([
                    production[:-1], net_production[:-1], self.Net_Fixed_inj1[:-1], self.Net_Fixed_inj2[:-1]
                ]).T
                y = net_production[1:]
                n_splits = (int(len(X) / 2) - 1)
                tscv = TimeSeriesSplit(n_splits=n_splits)
                models = [LinearRegression(), BayesianRidge()]
                # models = [LinearRegression(), Lasso(alpha=0, max_iter=100, tol=0.0001),
                #         Ridge(), ElasticNet(),  Lars(),
                #         LassoLars(), OrthogonalMatchingPursuit(), BayesianRidge(),
                #         ARDRegression(), SGDRegressor(), PassiveAggressiveRegressor(),
                #         KernelRidge(), SVR(), NuSVR(), LinearSVR(),
                #         KNeighborsRegressor(n_neighbors=3),
                #         RadiusNeighborsRegressor(radius=10000),
                #         GaussianProcessRegressor(), MLPRegressor(hidden_layer_sizes=(60,))]
                f.write('==========================================================\n')
                f.write('Producer {}\n'.format(i + 1))
                for model in models:
                    r2, mse = self.forward_walk_and_ML(X, y, model)
                    f.write('model: {}\n'.format(type(model)))
                    f.write('Average r2: {}\n'.format(r2))
                    f.write('Average MSE: {}\n'.format(mse))
                    f.write('\n')
                f.write('\n')
                f.write('\n')
                f.write('\n')

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
        plt.close()

    def plot_net_production_vs_time(self):
        plt.figure()
        plt.plot(self.Time, self.net_production_by_producer.T)
        plot_helper(
            title='Total Production vs Time',
            xlabel='Time',
            ylabel='Net Production',
            legend=self.producer_names,
            save=True
        )
        plt.close()

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
            plt.close()

    def plot_net_production_vs_injector(self):
        net_injections = [self.Net_Fixed_inj1, self.Net_Fixed_inj2]
        for i in range(len(net_injections)):
            plt.figure()
            for net_production in self.net_production_by_producer:
                plt.plot(self.Net_Fixed_inj1, net_production)
            plot_helper(
                title='Injector {}'.format(i + 1),
                xlabel='Injection Rate',
                ylabel='Net Production',
                legend=self.producer_names,
                save=True
            )
            plt.close()

    def production_rate_predictions(self):
        pass

    def net_production_predictions(self):
        self.crm_predict_net_production()
        self.predict_ML_net_production()

model = CRM(filename)
model.fit_producers()
model.fit_net_productions()
model.plot_producers_vs_time()
model.plot_net_production_vs_time()
model.plot_producers_vs_injector()
model.plot_net_production_vs_injector()
model.fit_ML_production_rate()
model.net_production_predictions()
