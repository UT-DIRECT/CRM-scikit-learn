# From: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_bagging.py
"""Bagging meta-estimator."""

# Author: Gilles Louppe <g.louppe@gmail.com>
# License: BSD 3 clause

# BSD 3-Clause License
#
# Copyright (c) 2007-2021 The scikit-learn developers.
# All rights reserved.

# As of November 11th, 2021, the code in this files borrows/directly uses code
# from the file and author mentioned above. Please see the link for the
# original code.


import itertools
import numbers
import numpy as np
from warnings import warn

from joblib import Parallel

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble._bagging import _generate_indices, _partition_estimators
from sklearn.utils import check_random_state, column_or_1d, deprecated
from sklearn.utils import indices_to_mask
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, _check_sample_weight

from src.helpers.fixes import delayed
from src.helpers.tsboot import mb_bootstrap_indicies


MAX_INT = np.iinfo(np.int32).max


def _generate_ts_indices(random_state, bootstrap, n_population, block_size):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = mb_bootstrap_indicies(n_population, block_size)
    else:
        # FIXME: block bootstrap without replacement
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )

    return indices


def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    bootstrap_samples,
    n_features,
    n_samples,
    max_features,
    block_size,
):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(
        random_state, bootstrap_features, n_features, max_features
    )
    sample_indices = _generate_ts_indices(
        random_state, bootstrap_samples, n_samples, block_size
    )

    return feature_indices, sample_indices


def _parallel_build_estimators(
    n_estimators, ensemble, X, y, sample_weight, seeds, total_n_estimators, verbose
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    block_size = ensemble._block_size
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            block_size,
        )

        estimator.fit((X[indices])[:, features], y[indices])
        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


class MBBaggingRegressor(BaggingRegressor):
    """Base class for Bagging meta-estimator.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
            self,
            base_estimator=None,
            n_estimators=10,
            *,
            block_size=5,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            warm_start=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
        ):
            super().__init__(
                base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                bootstrap_features=bootstrap_features,
                oob_score=oob_score,
                warm_start=warm_start,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
            )
            self.block_size = block_size


    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            multi_output=True,
        )
        return self._fit(X, y, self.block_size, sample_weight=sample_weight)

    def _parallel_args(self):
        return {}

    def _fit(self, X, y, block_size=None, max_depth=None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.
        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate block_size
        if block_size is None:
            block_size = self.block_size
        elif not isinstance(block_size, numbers.Integral):
            block_size = d

        if not (0 < block_size <= X.shape[0]):
            raise ValueError("block_size must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._block_size = block_size

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features * self.n_features_in_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_in_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        else:
            return y

    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_estimators()`
            feature_indices, sample_indices = _generate_bagging_indices(
                seed,
                self.bootstrap_features,
                self.bootstrap,
                self.n_features_in_,
                self._n_samples,
                self._max_features,
                self._block_size,
            )

            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):
        """
        The subset of drawn samples for each base estimator.
        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.
        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        return [sample_indices for _, sample_indices in self._get_estimators_indices()]

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `n_features_` was deprecated in version 1.0 and will be "
        "removed in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_
