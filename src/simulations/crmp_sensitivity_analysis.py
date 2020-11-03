import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_crmp import injectors, producers, producer_names
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.models import model_namer
from src.models.crmp import CRMP


q_sensitivity_analysis_file = INPUTS['crmp']['q_sensitivity_analysis']
q_sensitivity_analysis_data = {
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
    X, y = production_rate_dataset(producers[i], *injectors)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1, shuffle=False
    )
    for p0 in p0s:
        crmp = CRMP(p0=p0)
        crmp = crmp.fit(X_train, y_train)
        y_hat = crmp.predict(X_train)
        r2, mse = fit_statistics(y_train, y_hat)
        q_sensitivity_analysis_data['Producer'].append(i + 1)
        q_sensitivity_analysis_data['Model'].append(model_namer(crmp))
        q_sensitivity_analysis_data['tau_initial'].append(p0[0])
        q_sensitivity_analysis_data['tau_final'].append(crmp.tau_)
        q_sensitivity_analysis_data['f1_initial'].append(p0[1])
        q_sensitivity_analysis_data['f1_final'].append(crmp.gains_[0])
        q_sensitivity_analysis_data['f2_initial'].append(p0[2])
        q_sensitivity_analysis_data['f2_final'].append(crmp.gains_[1])
        q_sensitivity_analysis_data['r2'].append(r2)
        q_sensitivity_analysis_data['MSE'].append(mse)

q_sensitivity_analysis_df = pd.DataFrame(q_sensitivity_analysis_data)
q_sensitivity_analysis_df.to_csv(q_sensitivity_analysis_file)
