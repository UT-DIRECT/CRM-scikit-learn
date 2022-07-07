import sys
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_consistent_length, check_array


def fit_statistics(y_hat, y, shutin=False):
    if shutin:
        y_hat, y = remove_shutins(y_hat, y)
    try:
        if len(y_hat) == 0:
            return (0, 0)
        elif len(y_hat) < 2:
            r2 = np.nan
        else:
            r2 = r2_score(y_hat, y)
    except:
        r2 = 0
    try:
        mse = mean_squared_error(y_hat, y)
    except:
        mse = sys.float_info.max
    return (r2, mse)


def remove_shutins(y_hat, y):
    shutins = np.where(y == 0)[0]
    y = np.delete(y, shutins)
    y_hat = np.delete(y_hat, shutins)
    return (y_hat, y)
