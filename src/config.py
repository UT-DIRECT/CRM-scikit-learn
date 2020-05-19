import yaml


def read_inputs():
    with open('inputs.yml') as f:
        inputs = yaml.load(f, Loader=yaml.Loader)
    return inputs

INPUTS = read_inputs()
