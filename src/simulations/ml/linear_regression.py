import numpy as np
import pandas as pd

import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import INPUTS
from src.helpers.analysis import fit_statistics

from src.helpers.features import (
    get_real_producer_data, production_rate_dataset, producer_rows_from_df,
    construct_injection_rate_columns, construct_real_production_rate_dataset,
    construct_real_target_vector
)
from src.simulations import injector_names, producer_names


injector_data_file = INPUTS['real_data']['injector']
producer_data_file = INPUTS['real_data']['producer']
injectors_df = pd.read_csv(injector_data_file)
producers_df = pd.read_csv(producer_data_file)

injectors_df['Date'] = pd.to_datetime(injectors_df['Date'])
producers_df['Date'] = pd.to_datetime(producers_df['Date'])


def evaluate_linear_regression_model():
    for i in range(len(producer_names)):
        name = producer_names[i]
        print('Producer Name: ', name)
        producer = get_real_producer_data(producers_df, name, bhp=True)
        producer['On-Line'] = producers_df.loc[producers_df['Name'] == name, 'On-Line']
        producer['Av BHP'] = producers_df.loc[producers_df['Name'] == name, 'Av BHP']
        producer['Av WHP'] = (producers_df.loc[producers_df['Name'] == name, 'Av WHP'] + 1.013) * 14.503773773
        producer['Drawdown'] = producer['Av WHP'] - producer['Av BHP']
        injectors = injectors_df[['Name', 'Date', 'Water Vol']]
        df = construct_injection_rate_columns(producer, injectors)
        df['Time'] = (df['Date'] - df['Date'][0]).astype(int) / 8.64e13
        df.fillna(0, inplace=True)
        df.drop(columns=['Date'], inplace=True)
        X = df.iloc[:-1]
        y = producer[name].iloc[1:]
        print(X.head())
        print(y.head())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.7, shuffle=False
        )
        # model = linear_model.LinearRegression()
        model = ensemble.RandomForestRegressor()
        model = model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        # parameters = {
        #     'alpha': np.linspace(0, 1, 11),
        #     # 'l1_ratio': np.linspace(0, 1, 11),
        # }
        # gcv = GridSearchCV(model, parameters)
        # gcv.fit(X_train, y_train)
        # y_hat = gcv.predict(X_test)
        r2, mse = fit_statistics(y_hat, y_test)
        print('r2: ', r2)
        print('MSE: ', mse)
        break


evaluate_linear_regression_model()
