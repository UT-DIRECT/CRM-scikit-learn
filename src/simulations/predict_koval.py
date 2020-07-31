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

X = W_t[:-1]
y = f_w[1:]
train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
    X, y, 2
)
r2, mse, y_hat, time_step = test_model(X, y, koval, test_split)

time_test = time[-len(y_hat):]

koval_predictions_file = INPUTS['wfsim']['koval_predictions']
koval_predictions = {
    'Time': time_test,
    'Prediction': y_hat
}
koval_predictions_df = pd.DataFrame(koval_predictions)
koval_predictions_df.to_csv(koval_predictions_file)
