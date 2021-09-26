import pandas as pd
import yaml

from src.config import INPUTS


producer_names = ['PA01', 'PA02', 'PA03', 'PA05', 'PA09', 'PA10', 'PA12']
injector_names = ['IA04', 'IA08', 'IA11', 'IA13']

producer_data_file = INPUTS['real_data']['producer']
injector_data_file = INPUTS['real_data']['injector']
producers_df = pd.read_csv(producer_data_file)
injectors_df = pd.read_csv(injector_data_file)
