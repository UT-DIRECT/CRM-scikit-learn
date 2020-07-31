import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import f_w,  time, W_t
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models, test_model
from src.models.koval import Koval


# Loading the previously serialized Koval model
koval = load_models('koval')['koval']

step_size = 2
X = W_t[:-1]
y = f_w[1:]
train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
    X, y, step_size
)
r2, mse, y_hat, time_step = test_model(X, y, koval, test_split)

koval_predictions_file = INPUTS['wfsim']['koval_predictions']
koval_predictions = {
    'Step size': [],
    't_start': [],
    't_end': [],
    't_i': [],
    'Prediction': []
}
for i in range(len(y_hat)):
    y_hat_i = y_hat[i]
    time_step_i = time_step[i]
    t_start = time_step_i[0]
    t_end = time_step_i[-1]
    for k in range(len(y_hat_i)):
        y_i = y_hat_i[k]
        t_i = time_step_i[k]
        koval_predictions['Step size'].append(step_size)
        koval_predictions['t_start'].append(t_start)
        koval_predictions['t_end'].append(t_end)
        koval_predictions['t_i'].append(t_i)
        koval_predictions['Prediction'].append(y_i)

koval_predictions_df = pd.DataFrame(koval_predictions)
koval_predictions_df.to_csv(koval_predictions_file)
