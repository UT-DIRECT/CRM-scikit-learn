from sklearn.model_selection import train_test_split

from src.data.read_wfsim import qw_tank
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models
from src.models.koval import Koval


X, y = production_rate_dataset(qw_tank)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=False
)

# Loading the previously serialized Koval model
koval = load_models('koval')['koval']

y_hat = koval.predict(X_test)

r2, mse = fit_statistics(y_hat, y_test)
mse = mse / len(y_hat)

# TODO: Output these results to a CSV file
