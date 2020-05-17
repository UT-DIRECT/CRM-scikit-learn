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
