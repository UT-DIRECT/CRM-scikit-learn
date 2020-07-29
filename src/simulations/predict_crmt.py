from sklearn.model_selection import train_test_split

from src.data.read_wfsim import (delta_time, qo_tank, w_tank, qw_tank, q_tank)
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models
from src.models.crmt import CRMT


X, y = production_rate_dataset(q_tank, delta_time, w_tank)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=False
)


# Loading the previously serialized models
crmt = load_models('crmt')['crmt']

y_hat = crmt.predict(X_test)

r2, mse = fit_statistics(y_hat, y_test)
mse = mse / len(y_hat)

# TODO: Output these results to a CSV file
