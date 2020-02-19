import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics  import r2_score
from sklearn.model_selection import TimeSeriesSplit

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
        # X = q1, inj1, inj2
        self.q2 = lambda X, f1, f2, tau: X[0] * np.exp(-1 / tau) + (1 - np.exp(-1 / tau)) * (X[1] * f1 + X[2] * f2)
        self.p0 = .5, .5, 5

    def fit_producer(self, xdata, ydata):
        return curve_fit(
            self.q2, xdata, ydata, p0=self.p0,
            bounds=(
                (0.01, 0.01, 1,),
                (0.8, 0.8, 30)
            )
        )[0]

    def fit_producers(self):
        t = self.Time[1:]
        for i in range(len(self.producers)):
            producer = self.producers[i]
            xdata = np.array([
                producer[:-1], self.Fixed_inj1[:-1], self.Fixed_inj2[:-1]
            ])
            ydata = producer[1:]
            [f1, f2, tau] = self.fit_producer(xdata, ydata)
            q2 = self.q2(xdata, f1, f2, tau)
            plt.scatter(t, ydata)
            plt.plot(t, q2, 'r')
            plt.xlabel('Time')
            plt.ylabel('Production Rate')
            plt.title('Producer {}'.format(i + 1))
            plt.show()

    def fit_ML(self):
        producer = self.net_production_by_producer[0]
        xdata = np.array([
            producer[:-1], self.Net_Fixed_inj1[:-1], self.Net_Fixed_inj2[:-1]
        ]).T
        ydata = producer[1:]
        n_splits = (int(len(xdata) / 2) - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        r2_sum = 0
        model = LinearRegression()
        for train_index, test_index in tscv.split(xdata):
            x_train, x_test = xdata[train_index], xdata[test_index]
            y_train, y_test = ydata[train_index], ydata[test_index]
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            r2_sum += r2_score(y_predict, y_test)
        print('Average r2: {}'.format(r2_sum / n_splits))

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
            plt.scatter(self.fixed_inj1, producer)
        plt.title('injector 1')
        plt.xlabel('injection rate')
        plt.ylabel('production rate')
        plt.legend(['producer 1', 'producer 2', 'producer 3', 'producer 4'])
        plt.show()
        for producer in self.producers:
            plt.scatter(self.fixed_inj2, producer)
        plt.title('injector 2')
        plt.xlabel('injection rate')
        plt.ylabel('production rate')
        plt.legend(['producer 1', 'producer 2', 'producer 3', 'producer 4'])
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
# model.plot_producers_vs_time()
# model.plot_net_production_vs_time()
# model.plot_producers_vs_injector()
# model.plot_net_production_vs_injector()
model.fit_ML()
