from copy import deepcopy

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

from crmp import CRMP

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import goodness_score
from src.helpers.features import (
    production_rate_dataset, production_rate_features
)
from src.helpers.figures import plot_helper
from src.simulations import injector_names, number_of_producers


FIG_DIR = INPUTS['crmp']['figures_dir']

# References:
# https://medium.com/analytics-vidhya/3-time-series-forecasting-using-lstm-e14b93f4ec7c
# https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/

i = 3
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
X_train = np.array(X_train)

X_train_scaled = deepcopy(X_train)
y_train_scaled = deepcopy(y_train)
X_test_scaled = deepcopy(X_test)
y_test_scaled = deepcopy(y_test)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))


# Build The Model
activation = 'tanh'
optimizer = 'adam'
epoch = 10
batch = 128
input_layer = 128
layer_1 = 128
layer_2 = 128
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

history = model.fit(
    X_train_scaled, y_train_scaled, epochs=epoch,
    batch_size=batch, validation_split=0.1,
    verbose=0
)


y_hat_lstm = []
for j in range(30):
    y_hat_j = model.predict(X_test_scaled[j:(j + 1)])[0][0]
    X_test_scaled[j + 1] = y_hat_j
    y_hat_lstm.append(y_hat_j)

y_hat_lstm = np.array(y_hat_lstm).reshape(-1, 1)
y_hat_lstm = scaler.inverse_transform(y_hat_lstm)
r2, mse = fit_statistics(y_hat_lstm, y_test[:30])
print(mse)

crmp = CRMP().fit(X_train, y_train)
y_hat_crmp = crmp.predict(X_test[:30, 1:])
r2, mse = fit_statistics(y_hat_crmp, y_test[:30])
print(mse)

t = np.linspace(76, 105, 30)
plt.plot(t, y_test[:30], color='k', label='True Value', linewidth=2)
plt.plot(t, y_hat_crmp, alpha=0.5, label='CRMP', linewidth=2)
plt.plot(t, y_hat_lstm, alpha=0.5, label='LSTM Neural Network', linewidth=2)
plt.tight_layout()
plot_helper(
    FIG_DIR,
    title='{}: 30 Days Prediction for CRMP and LSTM Neural Network'.format(name),
    xlabel='Days',
    ylabel='Production Rate [bbls/day]',
    legend=True,
    save=True
)
