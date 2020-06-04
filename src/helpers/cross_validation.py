import numpy as np

from sklearn.linear_model import (BayesianRidge, ElasticNet, ElasticNetCV,
        Lasso, LassoCV, LinearRegression)

from .analysis import fit_statistics
from .models import test_model


TRAINING_SPLIT = 0.8


def forward_walk_and_ML(X, y, step_size, model):
    train_test_splits = forward_walk_splitter(X, y, step_size)
    return train_and_test_model(X, y, model, train_test_splits)


def forward_walk_splitter(X, y, step_size=2):
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
    train_test_seperation_idx = (int(TRAINING_SPLIT * len(split)) + 1)
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
