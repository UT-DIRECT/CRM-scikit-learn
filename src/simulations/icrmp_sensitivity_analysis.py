import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_crmp import (injectors, net_productions, producers,
         producer_names)
from src.helpers.analysis import fit_statistics
from src.helpers.features import net_production_dataset
from src.helpers.models import model_namer
from src.models.icrmp import ICRMP


N_sensitivity_analysis_file = INPUTS['crmp']['N_sensitivity_analysis']
N_sensitivity_analysis_data = {
    'Producer': [], 'Model': [], 'tau_initial': [], 'tau_final': [],
    'f1_initial': [], 'f1_final': [], 'f2_initial': [], 'f2_final': [],
    'r2': [], 'MSE': []
}

f1 = np.linspace(0, 1, 6)
f2 = np.ones(6) - f1
tau = np.linspace(1, 100, 10)
p0s = []
for i in tau:
    for j in range(len(f1)):
        p0s.append([i, f1[j], f2[j]])

for i in range(len(producers)):
    X, y = net_production_dataset(net_productions[i], producers[i], *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1, shuffle=False
    )
    for p0 in p0s:
        icrmp = ICRMP(p0=p0)
        icrmp = icrmp.fit(X_train, y_train)
        y_hat = icrmp.predict(X_train)
        r2, mse = fit_statistics(y_train, y_hat)
        N_sensitivity_analysis_data['Producer'].append(i + 1)
        N_sensitivity_analysis_data['Model'].append(model_namer(icrmp))
        N_sensitivity_analysis_data['tau_initial'].append(p0[0])
        N_sensitivity_analysis_data['tau_final'].append(icrmp.tau_)
        N_sensitivity_analysis_data['f1_initial'].append(p0[1])
        N_sensitivity_analysis_data['f1_final'].append(icrmp.gains_[0])
        N_sensitivity_analysis_data['f2_initial'].append(p0[2])
        N_sensitivity_analysis_data['f2_final'].append(icrmp.gains_[1])
        N_sensitivity_analysis_data['r2'].append(r2)
        N_sensitivity_analysis_data['MSE'].append(mse)

N_sensitivity_analysis_df = pd.DataFrame(N_sensitivity_analysis_data)
N_sensitivity_analysis_df.to_csv(N_sensitivity_analysis_file)
