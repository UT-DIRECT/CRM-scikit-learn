import pandas as pd
import yaml

from src.config import INPUTS


data_file = INPUTS['real_data']['data']
data = pd.read_csv(data_file)

producer_names = ['PA01', 'PA02', 'PA03', 'PA05', 'PA09', 'PA10', 'PA12']
producer_starting_indicies = [0, 160, 279, 433, 821, 853, 1074]
injector_names = ['IA04', 'IA08', 'IA11', 'IA13']

time = data['Time']
producers = [
    data[name + ' Total, bbls/day']
    for name in producer_names
]
producers_water_production = [
    data[name + ' Water, bbls/day']
    for name in producer_names
]
producers_oil_production = [
    data[name + ' Oil, bbls/day']
    for name in producer_names
]
injectors = [data[name] for name in injector_names]
