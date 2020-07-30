import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import (delta_time, q_tank, time, w_tank)
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models
from src.models.crmt import CRMT


X, y = production_rate_dataset(q_tank, delta_time, w_tank)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=False
)

# Loading the previously serialized CRMT model
crmt = load_models('crmt')['crmt']

y_hat = crmt.predict(X_test)
time_test = time[len(X_train) + 1:]

r2, mse = fit_statistics(y_hat, y_test)
mse = mse / len(y_hat)

crmt_predictions_file = INPUTS['wfsim']['crmt_predictions']
crmt_predictions = {
    'Time': time_test,
    'Prediction': y_hat
}
crmt_predictions_df = pd.DataFrame(crmt_predictions)
crmt_predictions_df.to_csv(crmt_predictions_file)
