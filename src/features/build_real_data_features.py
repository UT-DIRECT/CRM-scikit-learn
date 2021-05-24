import numpy as np
import pandas as pd

from src.config import INPUTS


raw_data_file = INPUTS['real_data']['raw_data']
processed_data_file = INPUTS['real_data']['data']

dfs = pd.read_excel(raw_data_file, sheet_name=[2, 3, 4])
dfs_list = list(dfs.values())

injection_df = dfs_list[0]
injection_df = injection_df.rename(columns={'Unnamed: 0': 'Time'})

oil_production_df, water_production_df = dfs_list[1:]
oil_production_df = oil_production_df.drop(columns=['Unnamed: 0'])
water_production_df = water_production_df.drop(columns=['Unnamed: 0'])
columns = list(oil_production_df.columns)
production_df = oil_production_df.add(water_production_df, fill_value=0)

injector_names = list(injection_df.columns)
producer_names = list(production_df.columns)
for name in injector_names:
    if name in producer_names:
        injector_name = name[1:]
        producer_name = name[0] + name[2:]
        injection_df = injection_df.rename(columns={name: injector_name})
        production_df = production_df.rename(columns={name: producer_name})

reservoir_df = pd.concat([injection_df, production_df], axis=1)
reservoir_df.to_csv(processed_data_file, index=False)
