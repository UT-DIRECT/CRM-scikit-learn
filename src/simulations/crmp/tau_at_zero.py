import numpy as np
import pandas as pd

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.features import production_rate_dataset
from src.models.crmp import CRMP


tau_at_zero_file = INPUTS['crmp']['crmp']['tau_at_zero']

initial_production_rate = producers[0][0]
fixed_injector_1 = 400
f1 = 0.3
fixed_injector_2 = 600
f2 = 0.4
taus = [1e-6, 1, 10, 20, 50, 100]

tau_at_zero_data = {
    'time': np.linspace(1, 150, 150).tolist(), 'tau_0': [], 'tau_1': [],
    'tau_10': [], 'tau_20': [], 'tau_50': [], 'tau_100': []
}


def constant_injection_producer():
    X, y = production_rate_dataset(producers[0], *injectors)
    crmp = CRMP().fit(X, y)
    crmp.gains_ = [f1, f2]
    for i in taus:
        crmp.tau_ = i
        q = [initial_production_rate]
        for k in range(150):
            X = np.array([q[-1], fixed_injector_1, fixed_injector_2])
            q.append(crmp.predict(X))
        if i < 1e-05:
            i = 0
        tau_at_zero_data['tau_{}'.format(i)] = q[1:]


constant_injection_producer()
tau_at_zero_df = pd.DataFrame(tau_at_zero_data)
tau_at_zero_df.to_csv(tau_at_zero_file)
