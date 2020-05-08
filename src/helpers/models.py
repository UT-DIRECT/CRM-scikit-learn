TRAINED_MODEL_DIR = './models'


def serialized_model_file(producer_name, model):
    # Removes the parameters found in the model name
    model_name = str(model)[:str(model).index('(')]
    return '{}/{}_{}.pkl'.format(
        TRAINED_MODEL_DIR,
        producer_name,
        model_name
    ).lower().replace(' ', '_')
