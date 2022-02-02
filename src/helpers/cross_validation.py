import numpy as np

from sklearn.linear_model import (BayesianRidge, ElasticNet, ElasticNetCV,
        Lasso, LassoCV, LinearRegression)

from src.helpers.analysis import fit_statistics, mean_absolute_percentage_error
from src.helpers.models import test_model


TRAINING_SPLIT = 0.8


def forward_walk_and_ML(X, y, step_size, model, training_split=TRAINING_SPLIT):
    train_test_splits = forward_walk_splitter(X, y, step_size, training_split=training_split)
    return train_and_test_model(X, y, model, train_test_splits)


def forward_walk_splitter(X, y, step_size, training_split=TRAINING_SPLIT):
    length = len(X)
    split = []
    for i in range(length - 1):
        train_end = i
        test_start = train_end + 1
        # test_start + step_size is one too long, so subtract by 1.
        # This is because linspace is inclusive.
        if test_start + step_size < length:
            test_end = int(test_start + step_size - 1)
        else:
            break
        train = np.linspace(0, train_end, num=(train_end + 1)).astype(int)
        # Sometimes the testing set is smaller than the step_size.
        # This is to ensure that there are the same number of splits for
        # all step_sizes.
        test = np.linspace(
            test_start, test_end, num=(test_end - test_start + 1)
        ).astype(int)
        split.append([train, test])
    train_test_seperation_idx = (int(training_split * len(split)) + 1)
    train_split = split[:train_test_seperation_idx]
    test_split = split[train_test_seperation_idx:]
    return (train_split, test_split, train_test_seperation_idx)


def train_and_test_model(X, y, model, train_test_splits):
    train_split = train_test_splits[0]
    test_split = train_test_splits[1]
    train_test_seperation_idx = train_test_splits[2]
    if isinstance(model, LinearRegression) or isinstance(model, BayesianRidge):
        model.fit(X[:train_test_seperation_idx], y[:train_test_seperation_idx])
    else:
        model = train_model_with_cv(X, y, model, train_split)
    return test_model(X, y, model, test_split)


def train_model_with_cv(X, y, model, train_split):
    fitted_model = model(cv=train_split, random_state=0).fit(X, y)
    if isinstance(fitted_model, LassoCV):
        trained_model = Lasso(alpha=fitted_model.alpha_)
    if isinstance(fitted_model, ElasticNetCV):
        trained_model = ElasticNet(
            alpha=fitted_model.alpha_, l1_ratio=fitted_model.l1_ratio_
        )
    return trained_model


def percentiles_of_interval(interval):
    p_lower = 50 - interval / 2.
    p_upper = 50 + interval / 2.
    return (p_lower, p_upper)


def confidence_interval_bounds(interval, data):
    p_lower, p_upper = percentiles_of_interval(interval)
    lower = np.percentile(data, p_lower)
    upper = np.percentile(data, p_upper)
    return (lower, upper)


def indicator_for_interval(y_hats, y_test, interval):
    counter = 0
    length = len(y_test)
    p_lower, p_upper = percentiles_of_interval(interval)
    for i in range(length):
        # y_hat_i = np.array([y_hat[i] for y_hat in y_hats])
        y_hat_i = y_hats[:, i]
        lower = np.percentile(y_hat_i, p_lower)
        upper = np.percentile(y_hat_i, p_upper)
        if lower < y_test[i] <= upper:
            counter += 1
    return counter / length


def average_indicator(y_hats, y_test, intervals):
    intervals = 100 * intervals
    average_indicator_values = [
        indicator_for_interval(y_hats, y_test, interval)
        for interval in intervals
    ]
    average_indicator_values[0] = 0.
    average_indicator_values[-1] = 1.
    return average_indicator_values


def accuracy_score(average_indicator_values, intervals):
    accuracies = []
    for indicator, interval in zip(average_indicator_values, intervals):
        if indicator >= interval:
            accuracies.append(1)
        else:
            accuracies.append(0)

    return accuracies


def precision_score(accuracy, average_indicator_values, intervals):
    precision = 1
    summation = 0
    for a, indicator, interval in zip(
            accuracy, average_indicator_values, intervals
    ):
        summation += a * (indicator - interval)

    precision -= 2 * summation / len(accuracy)
    return precision


def goodness_score(y_true, y_hats):
    intervals = np.linspace(0, 1, 11)
    average_indicator_values = average_indicator(
        y_hats, y_true, intervals
    )

    accuracies = accuracy_score(average_indicator_values, intervals)
    goodness = 1
    summation = 0
    for a, indicator, interval in zip(
            accuracies, average_indicator_values, intervals
    ):
        summation += (3 * a - 2) * (indicator - interval)

    goodness -= summation / len(accuracies)
    return goodness


# From: https://johnfoster.pge.utexas.edu/blog/posts/hackathon/
def scorer_for_crmp(estimator, X, y):
    l = len(X)
    estimator.q0 = X[0, 0]
    y_hat = estimator.predict(X[:, 1:])
    y_hats = []
    for e in estimator.estimators_:
        e.q0 = X[0, 0]
        y_hat_i = e.predict(X[:, 1:])
        y_hats.append(y_hat_i)
    y_hats = np.asarray(y_hats)
    mape = 1 - mean_absolute_percentage_error(y, y_hat)
    goodness = goodness_score(y, y_hats)
    score = 0.5 * mape + 0.5 * goodness
    return score
