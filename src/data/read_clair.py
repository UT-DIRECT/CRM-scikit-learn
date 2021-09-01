import pandas as pd
import yaml

from src.config import INPUTS


data_file = INPUTS['real_data']['data']
data = pd.read_csv(data_file)

producer_names = ['PA01', 'PA02', 'PA03', 'PA05', 'PA09', 'PA10', 'PA12']
producer_starting_indicies = [0, 160, 279, 433, 821, 853, 1074]
injector_names = ['IA04', 'IA08', 'IA11', 'IA13']

time = data['Time']
producers = [data[producer] for producer in producer_names]
injectors = [data[injector] for injector in injector_names]
