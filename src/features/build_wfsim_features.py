import pandas as pd

from src.config import INPUTS


def delta_time_calculator(time):
    delta_time = [0]
    for i in range(1, len(time)):
        delta_time.append(time[i] - time[i -  1])
    return delta_time

raw_data_file = INPUTS['wfsim']['raw_data']
processed_data_file = INPUTS['wfsim']['data']

df = pd.read_excel(raw_data_file, sheet_name=0, skiprows=2)

df = df.drop(df.columns[[0, 3, 5, 7, 9, 10, 11, 12]], axis=1)
df = df.rename(columns={'Unnamed: 1': 'time'}, errors='raise')
df = df.dropna()

df.insert(1, "Delta_time", delta_time_calculator(df['time']))

df.to_csv(processed_data_file, index=False)
