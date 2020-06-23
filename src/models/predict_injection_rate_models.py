import pandas as pd

from src.config import INPUTS
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import net_production_dataset, production_rate_dataset
from src.helpers.models import load_models, model_namer, test_model
from src.models import (injectors, net_productions, producers, producer_names,
    step_sizes)

# Loading the previously serialized models
trained_models = load_models()

injection_rate_models_by_producer = {}
for producer in producer_names:
    producer_label = producer.lower().replace(' ', '_')
    keys_for_producer = []
    for key in trained_models.keys():
        if producer_label in key and 'injectionratecrm' in key:
            keys_for_producer.append(key)
    injection_rate_models_by_producer[producer] = [trained_models[key] for key in keys_for_producer]


# Predict each producer with each model
I_predictions_file = INPUTS['files']['I_predictions']
net_I_predictions_file = INPUTS['files']['net_I_predictions']
I_predictions_data = {
    'Producer': [], 'Model': [], 't_i': [], 'q': []
}
net_I_predictions_data = {
    'Producer': [], 'Model': [], 'N': []
}
for i in range(len(producers)):
    producer_name = producer_names[i]
    producer_number = i + 1
    ircrm = injection_rate_models_by_producer[producer_name][0]
    X, y = production_rate_dataset(producers[i], *injectors)
    train_test_seperation_idx = forward_walk_splitter(X, y)[2]

    X_test = X[train_test_seperation_idx:]
    y_test = y[train_test_seperation_idx:]

    y_hat, injection_rates = ircrm.predict(X_test)
    for k in range(len(y_hat)):
        I_predictions_data['Producer'].append(producer_number)
        I_predictions_data['Model'].append(model_namer(ircrm))
        I_predictions_data['t_i'].append(train_test_seperation_idx + k + 1)
        I_predictions_data['q'].append(y_hat[i])

    net_I_predictions_data['Producer'].append(producer_number)
    net_I_predictions_data['Model'].append(model_namer(ircrm))
    net_I_predictions_data['N'].append(sum(y_hat))

    for j in range(len(injection_rates)):
        column_name = 'injector_{}'.format(j + 1)
        if column_name not in I_predictions_data:
            I_predictions_data[column_name] = injection_rates[j].tolist()
            net_I_predictions_data['Net_{}'.format(column_name)] = [sum(injection_rates[j])]
        else:
            for l in range(len(injection_rates[j])):
                I_predictions_data[column_name].append(injection_rates[j][l])
            net_I_predictions_data['Net_{}'.format(column_name)].append(sum(injection_rates[j]))


I_predictions_df = pd.DataFrame(I_predictions_data)
I_predictions_df.to_csv(I_predictions_file)
net_I_predictions_df = pd.DataFrame(net_I_predictions_data)
net_I_predictions_df.to_csv(net_I_predictions_file)
