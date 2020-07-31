import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import (delta_time, q_tank, time, w_tank)
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models, test_model
from src.models.crmt import CRMT


# Loading the previously serialized CRMT model
crmt = load_models('crmt')['crmt']

X, y = production_rate_dataset(q_tank, delta_time, w_tank)
train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
    X, y, 2
)
r2, mse, y_hat, time_step = test_model(X, y, crmt, test_split)

time_test = time[-len(y_hat):]

crmt_predictions_file = INPUTS['wfsim']['crmt_predictions']
crmt_predictions = {
    'Time': time_test,
    'Prediction': y_hat
}
crmt_predictions_df = pd.DataFrame(crmt_predictions)
crmt_predictions_df.to_csv(crmt_predictions_file)
