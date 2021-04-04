import pandas as pd
import yaml

from src.config import INPUTS


data_file = INPUTS['real_data']['data']
data = pd.read_csv(data_file)

producer_columns = ['PA01', 'PA02', 'PA03', 'PA05', 'PA09', 'PA10', 'PA12']
injector_columns = ['IA04', 'IA08', 'IA11', 'IA13']

time = data['Date']
producers = [data[column] for column in producer_columns]
injectors = [data[column] for column in injector_columns]
