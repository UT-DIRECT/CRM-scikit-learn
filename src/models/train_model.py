import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import *
from sklearn.metrics  import r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR

filename = "/Users/akhilpotla/ut/research/crm_validation/data/interim/CRMP_Corrected_July16_2018.csv"

class CRM():

    def __init__(self, filename):
        [
            self.Time, self.Random_inj1, self.Random_inj2, self.Fixed_inj1,
            self.Net_Fixed_inj1, self.Fixed_inj2, self.Net_Fixed_inj2,
            self.Prod1, self.Cum_Prod1, self.Prod2, self.Cum_Prod2,
            self.Prod3, self.Cum_Prod3, self.Prod4, self.Cum_Prod4
        ] = np.loadtxt(filename, delimiter=',', skiprows=1).T
        self.producers = np.array(
            [self.Prod1, self.Prod2, self.Prod3, self.Prod4]
        )
        self.net_production_by_producer = np.array(
            [self.Cum_Prod1, self.Cum_Prod2, self.Cum_Prod3, self.Cum_Prod4]
        )
        self.q2 = lambda X, f1, f2, tau: X[0] * np.exp(-1 / tau) + (1 - np.exp(-1 / tau)) * (X[1] * f1 + X[2] * f2)
        self.p0 = .5, .5, 5

    def fit_producer(self, X, y):
        return curve_fit(
            self.q2, X, y, p0=self.p0,
            bounds=(
                (0.01, 0.01, 1,),
                (0.8, 0.8, 30)
            )
        )[0]

    def fit_producers(self):
        t = self.Time[1:]
        for i in range(len(self.producers)):
            producer = self.producers[i]
            X = np.array([
                producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
            ])
            y = producer[1:]
            [f1, f2, tau] = self.fit_producer(X, y)
            q2 = self.q2(X, f1, f2, tau)
            plt.scatter(t, y)
            plt.plot(t, q2, 'r')
            plt.xlabel('Time')
            plt.ylabel('Production Rate')
            plt.title('Producer {}'.format(i + 1))
            plt.show()

    def crm_predict_net_production(self):
        with open('/Users/akhilpotla/ut/research/crm_validation/data/interim/crm_forward_walk_results.txt', 'w') as f:
            for i in range(len(self.producers)):
                producer = self.producers[i]
                X = np.array([
                    producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
                ]).T
                y = producer[1:]
                n_splits = (int(len(X) / 2) - 1)
                tscv = TimeSeriesSplit(n_splits=n_splits)
                r2_sum = 0
                for train_index, test_index in tscv.split(X):
                    x_train, x_test = (X[train_index]).T, (X[test_index]).T
                    y_train, y_test = y[train_index], y[test_index]
                    [f1, f2, tau] = self.fit_producer(x_train, y_train)
                    y_predict = self.q2(x_test, f1, f2, tau)
                    r2_sum += r2_score(y_predict, y_test)
                f.write('==========================================================\n')
                f.write('PRODUCER {}\n\n'.format(i + 1))
                f.write('f1: {}, f2: {}, tau: {}\n'.format(f1, f2, tau))
                f.write('Average r2: {}\n'.format(r2_sum / n_splits))
                f.write('\n\n\n\n')

    def fit_ML(self):
        net_production = self.net_production_by_producer[0]
        X = np.array([
            net_production[:-1], self.Net_Fixed_inj1[:-1], self.Net_Fixed_inj2[:-1]
        ]).T
        y = net_production[1:]
        print(X.shape)
        n_splits = (int(len(X) / 2) - 1)
        print(n_splits)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        r2_sum = 0
        # param_grid = {
        #         'alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0],
        #         'max_iter':[100, 1000, 10000],
        #         'tol':[0.0001, 0.01, 1, 100, 10000]
        #     }
        # model = Lasso()
        # grid = GridSearchCV(estimator=model, param_grid=param_grid)
        # grid.fit(X, y)
        # print('grid: ', grid)
        # print(grid.best_score_)
        # print(grid.best_estimator_)
        # models = [LinearRegression(), Lasso(alpha=0, max_iter=100, tol=0.0001),
        #         Ridge(), ElasticNet(),  Lars(),
        #         LassoLars(), OrthogonalMatchingPursuit(), BayesianRidge(),
        #         ARDRegression(), SGDRegressor(), PassiveAggressiveRegressor(),
        #         KernelRidge(), SVR(), NuSVR(), LinearSVR(),
        #         KNeighborsRegressor(n_neighbors=3),
        #         RadiusNeighborsRegressor(radius=10000),
        #         GaussianProcessRegressor(), MLPRegressor(hidden_layer_sizes=(60,))]
        with open('/Users/akhilpotla/ut/research/crm_validation/data/interim/train_output.txt', 'w') as f:
            for model in models:
                f.write('==========================================================\n')
                f.write('model: {}\n'.format(type(model)))
                for train_index, test_index in tscv.split(X):
                    x_train, x_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model.fit(x_train, y_train)
                    y_predict = model.predict(x_test)
                    r2_sum += r2_score(y_predict, y_test)
                f.write('Average r2: {}\n'.format(r2_sum / n_splits))
                f.write('\n')
                f.write('\n')
                f.write('\n')
                f.write('\n')

    def plot_producers_vs_time(self):
        plt.plot(self.Time, self.producers.T)
        plt.title('Production Rate vs Time')
        plt.xlabel('Time')
        plt.ylabel('Production Rate')
        plt.legend(['Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'])
        plt.show()

    def plot_net_production_vs_time(self):
        plt.plot(self.Time, self.net_production_by_producer.T)
        plt.title('Total Production vs Time')
        plt.xlabel('Time')
        plt.ylabel('Total Production')
        plt.legend(['Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'])
        plt.show()

    def plot_producers_vs_injector(self):
        for producer in self.producers:
            plt.scatter(self.Fixed_inj1, producer)
        plt.title('Injector 1')
        plt.xlabel('Injection Rate')
        plt.ylabel('Production Rate')
        plt.legend(['Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'])
        plt.show()
        for producer in self.producers:
            plt.scatter(self.Fixed_inj2, producer)
        plt.title('Injector 2')
        plt.xlabel('Injection Rate')
        plt.ylabel('Production Rate')
        plt.legend(['Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'])
        plt.show()

    def plot_net_production_vs_injector(self):
        for net_production in self.net_production_by_producer:
            plt.plot(self.Net_Fixed_inj1, net_production)
        plt.title('Injector 1')
        plt.xlabel('Injection Rate')
        plt.ylabel('Total Production')
        plt.legend(['Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'])
        plt.show()
        for net_production in self.net_production_by_producer:
            plt.plot(self.Net_Fixed_inj2, net_production)
        plt.title('Injector 2')
        plt.xlabel('Injection Rate')
        plt.ylabel('Production Rate')
        plt.legend(['Producer 1', 'Producer 2', 'Producer 3', 'Producer 4'])
        plt.show()

model = CRM(filename)
# model.fit_producers()
model.crm_predict_net_production()
# model.plot_producers_vs_time()
# model.plot_net_production_vs_time()
# model.plot_producers_vs_injector()
# model.plot_net_production_vs_injector()
# model.fit_ML()
