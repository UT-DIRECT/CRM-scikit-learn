import pickle
import dill as pickle

from sklearn.model_selection import train_test_split

from src.data.read_wfsim import (delta_time, qo_tank, w_tank, qw_tank, q_tank)
from src.helpers.features import production_rate_dataset
from src.helpers.models import serialized_model_path
from src.models.crmt import CRMT
from src.models.koval import Koval


X, y = production_rate_dataset(q_tank, delta_time, w_tank)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=False
)
crmt = CRMT().fit(X=X_train, y=y_train)

pickled_model = serialized_model_path('crmt', crmt)
with open(pickled_model, 'wb') as f:
    pickle.dump(crmt, f)
