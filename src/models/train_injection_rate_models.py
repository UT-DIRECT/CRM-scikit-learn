import pickle
import dill as pickle

import numpy as np

from src.helpers.cross_validation import (forward_walk_splitter,
        train_model_with_cv)
from src.helpers.features import net_production_dataset, production_rate_dataset
from src.helpers.models import serialized_model_path
from src.models import injectors, net_productions, producers, producer_names
from src.models.injection_rate_crm import InjectionRateCRM
