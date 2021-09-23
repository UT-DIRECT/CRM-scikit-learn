from datetime import datetime

import numpy as np
import pandas as pd

from src.config import INPUTS


raw_data_file = INPUTS['real_data']['raw_data']
producer_data_file = INPUTS['real_data']['producer']
injector_data_file = INPUTS['real_data']['injector']

producer_sheets = ['A01', 'A02', 'A03', 'A05', 'A09', 'A10', 'A12']
injector_sheets = ['A04', 'A08', 'A11', 'A13']


producers_df = pd.DataFrame()
for sheet_name in producer_sheets:
    producer_df = pd.read_excel(
        raw_data_file, sheet_name=sheet_name, skiprows=[1]
    )
    producer_df['Name'] = 'P' + sheet_name
    producer_df.set_index('Name', inplace=True)
    producers_df = producers_df.append(producer_df)

columns = [
    'Date', 'Oil Vol', 'Gas Vol', 'Water Vol', 'On-Line', 'Av BHP', 'Spot BHP',
    'Av BHT', 'Av WHP', 'total rate'
]
numerical_columns = columns[1:]
producers_df.fillna(0, inplace=True)
producers_df.drop(producers_df.columns.difference(columns), 1, inplace=True)
producers_df[numerical_columns] = producers_df[numerical_columns].clip(0)
producers_df['Total Vol'] = producers_df['Oil Vol'] + producers_df['Water Vol']


injectors_df = pd.DataFrame()
for sheet_name in injector_sheets:
    injector_df = pd.read_excel(
        raw_data_file, sheet_name=sheet_name, skiprows=[1]
    )
    injector_df['Name'] = 'I' + sheet_name
    injector_df.set_index('Name', inplace=True)
    injectors_df = injectors_df.append(injector_df)

columns = [
    'Date', 'Water Vol', 'On-Line', 'Av BHP', 'Spot BHP', 'Av BHT',
    'Av WHP', 'Cum WHP', 'Cum Volume'
]
numerical_columns = columns[1:]
injectors_df.fillna(0, inplace=True)
injectors_df.drop(injectors_df.columns.difference(columns), 1, inplace=True)
injectors_df[numerical_columns] = injectors_df[numerical_columns].clip(0)

producers_df.to_csv(producer_data_file)
injectors_df.to_csv(injector_data_file)
