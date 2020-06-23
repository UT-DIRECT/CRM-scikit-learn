import pickle
import dill as pickle

import numpy as np

from src.helpers.cross_validation import (forward_walk_splitter,
        train_model_with_cv)
from src.helpers.features import net_production_dataset, production_rate_dataset
from src.helpers.models import serialized_model_path
from src.models import injectors, net_productions, producers, producer_names
from src.models.injection_rate_crm import InjectionRateCRM


# Production Rate Training
for i in range(len(producers)):
    X, y = production_rate_dataset(producers[i], *injectors)
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(X, y)
    X_train = X[:train_test_seperation_idx]
    y_train = y[:train_test_seperation_idx]

    ircrm = InjectionRateCRM().fit(X_train, y_train)
    pickle_file = serialized_model_path(producer_names[i], ircrm)
    with open(pickle_file, 'wb') as f:
        pickle.dump(ircrm, f)
