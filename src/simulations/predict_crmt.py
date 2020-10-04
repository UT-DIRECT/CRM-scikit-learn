import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import (delta_time, qo_tank, w_tank, qw_tank, q_tank,
        time)
from src.helpers.analysis import fit_statistics
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models, model_namer, test_model
from src.models.crmt import CRMT
from src.simulations import step_sizes


# Loading the previously serialized models
trained_models = load_models('crmt')
crmt = trained_models['crmt']
del trained_models['crmt']
ml_models = list(trained_models.values())

crmt_predictions_file = INPUTS['wfsim']['crmt_predictions']
crmt_predictions_metrics_file = INPUTS['wfsim']['crmt_predictions_metrics']
crmt_predictions = {
    'Model': [], 'Step size': [], 't_start': [], 't_end': [], 't_i': [],
    'Prediction': []
}
crmt_predictions_metrics = {'Model': [], 'Step size': [], 'r2': [], 'MSE': []}


# CRMT Predictions
X, y = production_rate_dataset(q_tank, delta_time, w_tank)
for step_size in step_sizes:
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
        X, y, step_size
    )
    r2, mse, y_hat, time_step = test_model(X, y, crmt, test_split)
    crmt_predictions_metrics['Model'].append(model_namer(crmt))
    crmt_predictions_metrics['Step size'].append(step_size)
    crmt_predictions_metrics['r2'].append(r2)
    crmt_predictions_metrics['MSE'].append(mse)

    for i in range(len(y_hat)):
        y_hat_i = y_hat[i]
        time_step_i = time_step[i]
        t_start = time_step_i[0] + 2
        t_end = time_step_i[-1] + 2
        for k in range(len(y_hat_i)):
            y_i = y_hat_i[k]
            t_i = time_step_i[k] + 2
            crmt_predictions['Model'].append(model_namer(crmt))
            crmt_predictions['Step size'].append(step_size)
            crmt_predictions['t_start'].append(t_start)
            crmt_predictions['t_end'].append(t_end)
            crmt_predictions['t_i'].append(t_i)
            crmt_predictions['Prediction'].append(y_i)


# ML Model Predictions
X, y = production_rate_dataset(
    q_tank, time, qo_tank, w_tank, q_tank
)
for model in ml_models:
    for step_size in step_sizes:
        train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
            X, y, step_size
        )
        r2, mse, y_hat, time_step = test_model(X, y, model, test_split)
        crmt_predictions_metrics['Model'].append(model_namer(model))
        crmt_predictions_metrics['Step size'].append(step_size)
        crmt_predictions_metrics['r2'].append(r2)
        crmt_predictions_metrics['MSE'].append(mse)

        for i in range(len(y_hat)):
            y_hat_i = y_hat[i]
            time_step_i = time_step[i]
            t_start = time_step_i[0] + 2
            t_end = time_step_i[-1] + 2
            for k in range(len(y_hat_i)):
                y_i = y_hat_i[k]
                t_i = time_step_i[k] + 2
                crmt_predictions['Model'].append(model_namer(model))
                crmt_predictions['Step size'].append(step_size)
                crmt_predictions['t_start'].append(t_start)
                crmt_predictions['t_end'].append(t_end)
                crmt_predictions['t_i'].append(t_i)
                crmt_predictions['Prediction'].append(y_i)

crmt_predictions_df = pd.DataFrame(crmt_predictions)
crmt_predictions_metrics_df = pd.DataFrame(crmt_predictions_metrics)
crmt_predictions_df.to_csv(crmt_predictions_file)
crmt_predictions_metrics_df.to_csv(crmt_predictions_metrics_file)
