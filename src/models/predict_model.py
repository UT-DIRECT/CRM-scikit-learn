import pickle
import dill as pickle

from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import (load_models, model_namer,
    serialized_model_path, test_model)
from src.models import injectors, producers, producer_names, step_sizes
from src.models.crm import CRM


# Loading the previously serialized models
models = load_models()

# Splitting the models up by producer
models_by_producer = {}
for producer in producer_names:
    producer_label = producer.lower().replace(' ', '_')
    keys_for_producer = [key for key in models.keys() if producer_label in key]
    models_by_producer[producer] = [models[key] for key in keys_for_producer]

# Predict each producer with each model
for i in range(len(producers)):
    producer_name = producer_names[i]
    models = models_by_producer[producer_name]
    X, y = production_rate_dataset(producers[i], *injectors)

    for model in models:
        for step_size in step_sizes:
            test_split = forward_walk_splitter(X, y, step_size)[1]
            r2, mse, y_hat = test_model(X, y, model, test_split)
