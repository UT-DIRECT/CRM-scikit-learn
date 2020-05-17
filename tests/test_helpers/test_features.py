from unittest.mock import patch

import numpy as np

from src.helpers.features import *


class TestProductionRateFeatures():


    def test_no_injectors(self):
        q = np.array([2, 3, 4, 5, 6])
        X = production_rate_features(q)
        assert(X.size == 4)


    def test_one_injector(self):
        q = np.array([2, 3, 4, 5, 6])
        inj1 = 2 * q
        X = production_rate_features(q, inj1)
        assert(X.shape == (4, 2))


    def test_three_injectors(self):
        q = np.array([2, 3, 4, 5, 6])
        inj1 = 2 * q
        inj2 = 3 * q
        inj3 = 4 * q
        X = production_rate_features(q, inj1, inj2, inj3)
        assert(X.shape == (4, 4))


    def test_producer_and_injectors_different_size(self):
        q = np.array([2, 3, 4, 5, 6])
        inj1 = np.array([2, 3, 4, 5])
        inj2 = np.array([2, 3, 4, 5, 6, 7, 8])
        X = production_rate_features(q, inj1, inj2)
        assert(X.shape == (4, 3))


class TestNetProductionFeatures():


    def test_net_production_and_production_rate_with_injector(self):
        N = np.array([2, 5, 9, 14, 20])
        q = np.array([2, 3, 4, 5, 6])
        inj1 = np.array([2, 3, 4, 5, 6])
        inj2 = np.array([2, 3, 4, 5, 6])
        X = net_production_features(N, q, inj1, inj2)
        assert(X.shape == (4, 4))


    def test_net_production_and_production_rate_without_injectors(self):
        N = np.array([2, 5, 9, 14, 20])
        q = np.array([2, 3, 4, 5, 6])
        X = net_production_features(N, q)
        assert(X.shape == (2, 4))


    def test_net_production_and_smaller_production_rate(self):
        N = np.array([2, 5, 9, 14, 20])
        q = np.array([2, 3, 4])
        inj1 = np.array([2, 3, 4, 5])
        inj2 = np.array([2, 3, 4, 5, 6, 7, 8])
        X = net_production_features(N, q, inj1, inj2)
        assert(X.shape == (4,))


    def test_net_production_and_larger_production_rate(self):
        N = np.array([2, 5, 9, 14, 20])
        q = np.array([2, 3, 4, 5, 6, 7, 8])
        inj1 = np.array([2, 3, 4, 5, 6])
        inj2 = np.array([2, 3, 4, 5, 6])
        X = net_production_features(N, q, inj1, inj2)
        assert(X.shape == (4, 4))


class TestTargetVector():


    def test_target_vector(self):
        y = np.array([2, 3, 4, 5])
        y = target_vector(y)
        assert(y.shape == (3,))


class TestProductionRateDataset():


    @patch('src.helpers.features.target_vector')
    @patch('src.helpers.features.production_rate_features')
    def test_production_rate_dataset(
        self, production_rate_features, target_vector
    ):
        q = np.array([2, 3, 4, 5, 6])
        inj1 = 2 * q
        production_rate_dataset(q, inj1)
        assert production_rate_features.called
        assert target_vector.called


class TestNetProductionDataset():


    @patch('src.helpers.features.target_vector')
    @patch('src.helpers.features.net_production_features')
    def test_net_production_dataset(
            self, net_production_features, target_vector
    ):
        q = np.array([2, 3, 4, 5, 6])
        N = 2 * q
        net_production_dataset(N, q)
        assert net_production_features.called
        assert target_vector.called
