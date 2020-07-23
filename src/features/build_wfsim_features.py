import pandas as pd

from src.config import INPUTS


raw_data_file = INPUTS['wfsim']['raw_data']
processed_data_file = INPUTS['wfsim']['data']

df = pd.read_excel(raw_data_file, sheet_name=0, skiprows=2)

df = df.drop(df.columns[[0, 3, 5, 7, 9, 10, 11, 12]], axis=1)
df = df.rename(columns={'Unnamed: 1': 'Time'}, errors='raise')
df = df.dropna()

df.to_csv(processed_data_file, index=False)
