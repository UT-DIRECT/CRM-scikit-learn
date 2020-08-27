from sklearn.linear_model import BayesianRidge

from src.helpers.models import serialized_model_path, model_namer


class TestSerializedModelFile():


    def test_serialized_model_path(self):
        model = BayesianRidge()
        model_path = serialized_model_path(
            'ml_models', model, producer_name='Producer 1'
        )
        assert(model_path == './models/ml_models/producer_1_bayesianridge.pkl')


class TestModelNamer():


    def test_model_namer(self):
        model = BayesianRidge()
        model_name = model_namer(model)
        assert(model_name == 'BayesianRidge')


# TODO: Test load_models
# TODO: Test test_model
