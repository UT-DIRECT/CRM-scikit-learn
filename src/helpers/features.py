import numpy as np


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
    sigma = column.std()
    gaussian_noise = np.random.normal(loc=0, scale=sigma, size=length)
    exponential_decline_scaling = np.linspace(0.1, 2, num=length)
    for i in range(length):
        column[i] += abs(gaussian_noise[i])
        column[i] *= 1 / exponential_decline_scaling[i]
        # Multiplying by 1/exponential_decline_scaling increases the value of
        # the dataset by 3; therefore we multiply by 1/3 below.
    return column * 1 / 3


def net_flow(production):
    net = []
    for prod in production:
        if len(net) == 0:
            net.append(prod)
        else:
            net.append(net[-1] + prod)
    return net
