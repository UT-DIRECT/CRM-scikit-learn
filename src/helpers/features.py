import numpy as np
import pandas as pd

from src.simulations import injector_names


def production_rate_features(q, *I):
    size = q[:-1].size
    if len(I) == 0:
        return np.array(q[:size])
    else:
        injectors = [i[:size] for i in I]
        return np.array([
            q[:size], *injectors
        ]).T


def net_production_features(N, q, *I):
    size = N.size - 1
    if len(I) == 0:
        return np.array([N[:size], q[:size]])
    else:
        injectors = [i[:size] for i in I]
        return np.array([
            N[:size], q[:size], *injectors
        ]).T


def target_vector(y):
    return y[1:]


def production_rate_dataset(q, *I):
    return [
        production_rate_features(q, *I),
        target_vector(q)
    ]


def get_real_producer_data(df, name, bhp=False):
    columns = ['Date', 'Total Vol']
    if bhp:
        columns.append('Av BHP')
    producer = df.loc[df['Name'] == name, columns]
    producer = construct_change_in_pressure_column(producer)
    producer[name] = producer['Total Vol']
    producer.drop(columns=['Total Vol', 'Av BHP'], inplace=True)
    df.fillna(0, inplace=True)
    return producer


def construct_real_production_rate_dataset(q, I, bhp):
    return [
        construct_real_production_rate_features(q, I, bhp),
        construct_real_target_vector(q)
    ]


def construct_real_target_vector(q):
    producer_name = q.columns[1]
    producer = q[producer_name][1:]
    return producer


def construct_column_of_length(data, length_of_column):
    if length_of_column > len(data):
        zeros = np.zeros(length_of_column - len(data))
        return np.append(zeros, data.to_numpy())
    else:
        return data[-length_of_column:]


def construct_real_production_rate_features(q, I, bhp):
    df = pd.DataFrame(q)
    df - construct_bhp_column(df, bhp)
    df = construct_injection_rate_columns(df, I)
    df.drop(columns=['Date'], inplace=True)
    df.fillna(0, inplace=True)
    return df.iloc[:-1]


def construct_bhp_column(df, bhp):
    if bhp is not None:
        df['delta_p'] = bhp
    return df


def construct_change_in_pressure_column(df):
    bhp = df['Av BHP']
    delta_p = bhp[:-1] - bhp[1:]
    df['delta_p'] = delta_p
    return df


def construct_injection_rate_columns(df, I):
    length = len(df.index)
    for name in injector_names:
        injector = I.loc[
            I['Name'] == name,
            'Water Vol'
        ]
        injector_column = construct_column_of_length(injector, length)
        df[name] = injector_column
    return df


def net_production_dataset(N, q, *I):
    return [
        net_production_features(N, q, *I),
        target_vector(N)
    ]


def koval_dataset(W_t, f_w):
    X = W_t[:-1]
    y = f_w[1:]
    return [
        X, y
    ]


def white_noise(column):
    length = len(column)
    rolling_stds = column.rolling(7).std()
    # sigma = column.std()
    # gaussian_noise = np.random.normal(loc=0, scale=sigma, size=length)
    # exponential_decline_scaling = np.linspace(0.1, 2, num=length)
    for i in range(length):
        sigma = rolling_stds[i]
        if np.isnan(sigma):
            continue
        gaussian_noise = np.random.normal(loc=0, scale=rolling_stds[i])
        column[i] += gaussian_noise
    #     if column[i] < 0:
    #         column[i] = 0
    #     # column[i] *= 1 / exponential_decline_scaling[i]
    #     # Multiplying by 1/exponential_decline_scaling increases the value of
    #     # the dataset by 3; therefore we multiply by 1/3 below.
    return column


def net_flow(production):
    net = []
    for prod in production:
        if len(net) == 0:
            net.append(prod)
        else:
            net.append(net[-1] + prod)
    return net


def producer_rows_from_df(df, producer):
    return df.loc[df['Producer'] == producer]
