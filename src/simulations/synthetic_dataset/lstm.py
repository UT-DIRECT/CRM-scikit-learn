import sys

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import goodness_score
from src.helpers.features import (
    production_rate_dataset, production_rate_features
)
from src.simulations import injector_names, number_of_producers


# References:
# https://medium.com/analytics-vidhya/3-time-series-forecasting-using-lstm-e14b93f4ec7c
# https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
results_output_file = INPUTS['crmp']['lstm']
results = {
    'Producer': [], 'Activation': [], 'Optimizer': [], 'Epoch': [], 'Batch': [],
    'Input Layer Units': [], 'Layer 1': [], 'Layer 2': [], 'r2': [], 'MSE': []
}

iteration = 0
for i in [3]:
    min_mse = sys.float_info.max
    producer = producers[i]
    producer += np.random.normal(loc=0.0, scale=25, size=len(producer))
    name = producer_names[i]
    print(name)

    activations = ['sigmoid', 'relu', 'elu','tanh']
    optimizers = ['adam', 'adagrad', 'sgd']
    epochs = [5, 10, 20]
    batches = [32, 64, 128]
    input_layers = [128, 256]
    layer_1s = [64, 128]
    layer_2s = [64, 128]

    X, y = production_rate_dataset(producer, *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5, shuffle=False
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = scaler.fit_transform(X_test)
    y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))


    for activation in activations:
        for optimizer in optimizers:
            for epoch in epochs:
                for batch in batches:
                    for input_layer in input_layers:
                        for layer_1 in layer_1s:
                            for layer_2 in layer_2s:
                                iteration += 1
                                percent_complete = iteration / 3456. * 100
                                print('Iteration: {}/3456; {:.1f}%'.format(
                                    iteration, percent_complete
                                ))
                                # Build The Model
                                model = Sequential()

                                model.add(LSTM(
                                    units=input_layer,
                                    activation=activation,
                                    return_sequences=True,
                                    input_shape=(X_train.shape[1],1)
                                ))
                                model.add(LSTM(
                                    units=layer_1,
                                    activation=activation,
                                    return_sequences=True
                                ))
                                model.add(LSTM(
                                    units=layer_2,
                                    activation=activation,
                                    return_sequences=False
                                ))
                                model.add(Dense(units=1)) # Prediction of the next value

                                model.compile(
                                    optimizer=optimizer,
                                    loss='mean_squared_error'
                                )
                                # model.summary()

                                history = model.fit(
                                    X_train_scaled, y_train_scaled, epochs=epoch,
                                    batch_size=batch, validation_split=0.1,
                                    verbose=0
                                )


                                y_hat = []
                                for j in range(30):
                                    y_hat_j = model.predict(X_test_scaled[j:(j + 1)])[0][0]
                                    X_test_scaled[j + 1] = y_hat_j
                                    y_hat.append(y_hat_j)

                                y_hat = np.array(y_hat).reshape(-1, 1)
                                try:
                                    y_hat = scaler.inverse_transform(y_hat)
                                    r2, mse = fit_statistics(y_hat, y_test[:30])
                                except:
                                    r2 = 0
                                    mse = sys.float_info.max
                                if mse < min_mse:
                                    min_mse = mse
                                results['Producer'].append(name)
                                results['Activation'].append(activation)
                                results['Optimizer'].append(optimizer)
                                results['Epoch'].append(epoch)
                                results['Batch'].append(batch)
                                results['Input Layer Units'].append(input_layer)
                                results['Layer 1'].append(layer_1)
                                results['Layer 2'].append(layer_2)
                                results['r2'].append(r2)
                                results['MSE'].append(mse)
    print(min_mse)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_output_file)
