import pickle
import dill as pickle

from sklearn.model_selection import train_test_split

from src.data.read_wfsim import (delta_time, qo_tank, w_tank, qw_tank, q_tank)
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import serialized_model_path
from src.models.crmt import CRMT


X, y = production_rate_dataset(q_tank, delta_time, w_tank)
train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
    X, y, 2
)
X_train = X[:train_test_seperation_idx]
y_train = y[:train_test_seperation_idx]
crmt = CRMT().fit(X=X_train, y=y_train)

pickled_model = serialized_model_path('crmt', crmt)
with open(pickled_model, 'wb') as f:
    pickle.dump(crmt, f)
