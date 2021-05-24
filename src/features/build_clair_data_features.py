import numpy as np
import pandas as pd

from src.config import INPUTS


raw_data_file = INPUTS['real_data']['raw_data']
processed_data_file = INPUTS['real_data']['data']

producer_sheets = ['A01', 'A02', 'A03', 'A05', 'A09', 'A10', 'A12']
injector_sheets = ['A04', 'A08', 'A11', 'A13']
column_names = []
df = pd.DataFrame()


def data_from_column(df, column_name):
    return df[column_name][1:]


def calculate_production_data(df):
    oil_data = data_from_column(df, 'Oil Vol')
    water_data = data_from_column(df, 'Water Vol')
    production_data = oil_data + water_data
    return production_data


def construct_column_of_length(data, length_of_column):
    zeros = np.zeros(length_of_column - len(data))
    return np.append(zeros, data.to_numpy())


for sheet_name in producer_sheets:
    producer_df = pd.read_excel(raw_data_file, sheet_name=sheet_name)
    if sheet_name == 'A01':
        length = len(producer_df['Date'][1:])
        df['Date'] = [producer_df['Date'][i + 1] for i in range(length)]
    production_data = calculate_production_data(producer_df)
    column_name = 'P' + sheet_name
    df[column_name] = construct_column_of_length(production_data, length)
    column_names.append(column_name)

for sheet_name in injector_sheets:
    injector_df = pd.read_excel(raw_data_file, sheet_name=sheet_name)
    water_data = injector_df['Water Vol'][1:]
    column_name = 'I' + sheet_name
    df[column_name] = construct_column_of_length(water_data, length)
    column_names.append(column_name)

df.fillna(0, inplace=True)
df[column_names] = df[column_names].clip(0)

df.to_csv(processed_data_file)
