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
i = 0
producer = producers[i]
name = producer_names[i]
print(name)

activation = 'relu'
optimizer = 'adam'
batch = 32
epoch = 50
units = 256

X, y = production_rate_dataset(producer, *injectors)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.5, shuffle=False
)
X_train = np.array(X_train)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))


# Build The Model
model = Sequential()

model.add(LSTM(
    units=256, activation='relu', return_sequences=True,
    input_shape=(X_train.shape[1],1)
))
model.add(Dropout(0.2))
model.add(LSTM(units=256, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next value

model.compile(optimizer='adam', loss='mean_squared_error')
# model.summary()

history = model.fit(
    X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.1
)


y_hat = []
for j in range(30):
    y_hat_j = model.predict(X_test_scaled[j:(j + 1)])[0][0]
    X_test_scaled[j + 1] = y_hat_j
    y_hat.append(y_hat_j)

y_hat = np.array(y_hat).reshape(-1, 1)
y_hat = scaler.inverse_transform(y_hat)
r2, mse = fit_statistics(y_hat, y_test[:30])
print('r2: ', r2)
print('mse: ', mse)
print()
print()
