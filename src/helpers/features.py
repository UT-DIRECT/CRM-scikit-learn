import numpy as np


def production_rate_features(q, *I):
    size = q[:-1].size
    if len(I) == 0:
        return np.array(q[:size])
    else:
        injectors = [i[:size] for i in I]
        return np.array([
            q[:size], *injectors
        ])


def net_production_features(N, q):
    size = N.size - 1
    return np.array([N[:size], q[:size]])


def target_vector(y):
    return y[1:]


def production_rate_dataset(q, *I):
    return [
        production_rate_features(q, *I).T,
        target_vector(q)
    ]


def net_production_dataset(N, q):
    return [
        net_production_features(N, q).T,
        target_vector(N)
    ]
