import numpy as np
import pandas as pd

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names, time
from src.helpers.features import production_rate_dataset
from src.models.crmp import CRMP


tau_at_zero_file = INPUTS['crmp']['crmp']['tau_at_zero']

initial_production_rate = producers[0][0]
fixed_injector_1 = 400
f1 = 0.3
fixed_injector_2 = 600
f2 = 0.4
tau = 1e-6


def constant_injection_producer():
    X, y = production_rate_dataset(producers[0], *injectors)
    crmp = CRMP().fit(X, y)
    crmp.tau_ = tau
    crmp.gains_ = [f1, f2]
    q = np.zeros(151)
    q[0] = initial_production_rate
    for i in time:
        X = np.array([q[-1], fixed_injector_1, fixed_injector_2])
        q[int(i)] = crmp.predict(X)
    df = pd.DataFrame({'time': time, 'q1': q[1:]})
    return df


tau_at_zero_df = constant_injection_producer()
tau_at_zero_df.to_csv(tau_at_zero_file)
