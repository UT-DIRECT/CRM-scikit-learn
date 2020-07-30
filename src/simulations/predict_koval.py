import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import INPUTS
from src.data.read_wfsim import f_w,  time, W_t
from src.helpers.analysis import fit_statistics
from src.helpers.features import production_rate_dataset
from src.helpers.models import load_models
from src.models.koval import Koval


X = W_t[:-1]
y = f_w[1:]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=False
)

# Loading the previously serialized Koval model
koval = load_models('koval')['koval']

y_hat = koval.predict(X_test)
time_test = time[len(X_train) + 1:]

r2, mse = fit_statistics(y_hat, y_test)
mse = mse / len(y_hat)


koval_predictions_file = INPUTS['wfsim']['koval_predictions']
koval_predictions = {
    'Time': time_test,
    'Prediction': y_hat
}
koval_predictions_df = pd.DataFrame(koval_predictions)
koval_predictions_df.to_csv(koval_predictions_file)
