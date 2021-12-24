import pandas as pd

from crmp import Koval
from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import f_w,  time, W_t
from src.helpers.cross_validation import forward_walk_splitter
from src.helpers.analysis import fit_statistics
from src.helpers.features import koval_dataset, production_rate_dataset
from src.helpers.models import load_models, model_namer, test_model
from src.simulations import step_sizes


# Loading the previously serialized Koval model
trained_models = load_models('koval')
koval = trained_models['koval']
del trained_models['koval']
ml_models = list(trained_models.values())

koval_predictions_file = INPUTS['wfsim']['koval_predictions']
koval_predictions_metrics_file = INPUTS['wfsim']['koval_predictions_metrics']
koval_predictions = {
    'Model': [], 'Step size': [], 't_start': [], 't_end': [], 't_i': [],
    'Prediction': []
}
koval_predictions_metrics = {'Model': [], 'Step size': [], 'r2': [], 'MSE': []}


# Koval Predictions
X, y = koval_dataset(W_t, f_w)
for step_size in step_sizes:
    train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
        X, y, step_size, training_split=0.8
    )
    r2, mse, y_hat, time_step = test_model(X, y, koval, test_split)

    koval_predictions_metrics['Model'].append(model_namer(koval))
    koval_predictions_metrics['Step size'].append(step_size)
    koval_predictions_metrics['r2'].append(r2)
    koval_predictions_metrics['MSE'].append(mse)

    for i in range(len(y_hat)):
        y_hat_i = y_hat[i]
        time_step_i = time_step[i]
        t_start = time_step_i[0] + 2
        t_end = time_step_i[-1] + 2
        for k in range(len(y_hat_i)):
            y_i = y_hat_i[k]
            t_i = time_step_i[k] + 2
            koval_predictions['Model'].append(model_namer(koval))
            koval_predictions['Step size'].append(step_size)
            koval_predictions['t_start'].append(t_start)
            koval_predictions['t_end'].append(t_end)
            koval_predictions['t_i'].append(t_i)
            koval_predictions['Prediction'].append(y_i)


# ML Model Predictions
X, y = production_rate_dataset(f_w, W_t)
for model in ml_models:
    for step_size in step_sizes:
        train_split, test_split, train_test_seperation_idx = forward_walk_splitter(
            X, y, step_size
        )
        r2, mse, y_hat, time_step = test_model(X, y, model, test_split)
        koval_predictions_metrics['Model'].append(model_namer(model))
        koval_predictions_metrics['Step size'].append(step_size)
        koval_predictions_metrics['r2'].append(r2)
        koval_predictions_metrics['MSE'].append(mse)

        for i in range(len(y_hat)):
            y_hat_i = y_hat[i]
            time_step_i = time_step[i]
            t_start = time_step_i[0] + 2
            t_end = time_step_i[-1] + 2
            for k in range(len(y_hat_i)):
                y_i = y_hat_i[k]
                t_i = time_step_i[k] + 2
                koval_predictions['Model'].append(model_namer(model))
                koval_predictions['Step size'].append(step_size)
                koval_predictions['t_start'].append(t_start)
                koval_predictions['t_end'].append(t_end)
                koval_predictions['t_i'].append(t_i)
                koval_predictions['Prediction'].append(y_i)


koval_predictions_df = pd.DataFrame(koval_predictions)
koval_predictions_metrics_df = pd.DataFrame(koval_predictions_metrics)
koval_predictions_df.to_csv(koval_predictions_file)
koval_predictions_metrics_df.to_csv(koval_predictions_metrics_file)
