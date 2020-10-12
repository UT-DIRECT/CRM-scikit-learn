import pickle
import dill as pickle


from sklearn.linear_model import (BayesianRidge, ElasticNetCV, LassoCV,
        LinearRegression)
from sklearn.model_selection import train_test_split

from src.data.read_wfsim import (delta_time, qo_tank, w_tank, qw_tank, q_tank,
        time)
from src.helpers.cross_validation import (forward_walk_splitter,
        train_model_with_cv)
from src.helpers.features import production_rate_dataset
from src.helpers.models import is_CV_model, serialized_model_path
from src.models.crmt import CRMT

X, y = production_rate_dataset(q_tank, delta_time, w_tank)
train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
    X, y, 2
)
X_train = X[:train_test_seperation_idx]
y_train = y[:train_test_seperation_idx]
model = CRMT().fit(X_train, y_train)
pickled_model = serialized_model_path('crmt', model)
with open(pickled_model, 'wb') as f:
    pickle.dump(model, f)


X, y = production_rate_dataset(
    q_tank, time, qo_tank, w_tank, q_tank
)
X_train = X[:train_test_seperation_idx]
y_train = y[:train_test_seperation_idx]

models = [
    BayesianRidge(), ElasticNetCV, LassoCV, LinearRegression()
]
for model in models:
    if is_CV_model(model):
        model = train_model_with_cv(X, y, model, train_split)
    model = model.fit(X_train, y_train)
    pickled_model = serialized_model_path('crmt', model)
    with open(pickled_model, 'wb') as f:
        pickle.dump(model, f)
