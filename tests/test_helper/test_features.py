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
        assert(X.shape == (2, 4))


    def test_three_injectors(self):
        q = np.array([2, 3, 4, 5, 6])
        inj1 = 2 * q
        inj2 = 3 * q
        inj3 = 4 * q
        X = production_rate_features(q, inj1, inj2, inj3)
        assert(X.shape == (4, 4))


    def test_producer_and_injectors_different_size(self):
        q = np.array([2, 3, 4, 5, 6])
        inj1 = np.array([2, 3, 4])
        inj2 = np.array([2, 3, 4, 5, 6, 7, 8])
        X = production_rate_features(q, inj1, inj2)
        assert(X.shape == (3,))
