import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

def fit_statistics(y_hat, y):
    if len(y_hat)  < 2:
        r2 = np.nan
    else:
        r2 = r2_score(y_hat, y)
    mse = mean_squared_error(y_hat, y)
    return (r2, mse)
