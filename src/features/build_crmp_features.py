import pandas as pd

from src.config import INPUTS
from src.helpers.features import net_flow, white_noise


raw_data_file = INPUTS['crmp']['raw_data']
processed_data_file = INPUTS['crmp']['data']

df = pd.read_excel(raw_data_file, 0, skiprows=9)
df = df.drop(df.columns[[1, 2, 3, 4, 7, 8]], axis=1)
df.columns = [
    'time', 'fixed_inj1', 'fixed_inj2',
    'Prod1', 'Prod2', 'Prod3', 'Prod4'
]
# columns = df.columns[2:]
# for column in columns:
#     df[column] = white_noise(df[column])


df.insert(2, 'net_fixed_inj1', net_flow(df['fixed_inj1']))
df.insert(4, 'net_fixed_inj2', net_flow(df['fixed_inj2']))
df.insert(6, 'Net_Prod1', net_flow(df['Prod1']))
df.insert(8, 'Net_Prod2', net_flow(df['Prod2']))
df.insert(10, 'Net_Prod3', net_flow(df['Prod3']))
df.insert(12, 'Net_Prod4', net_flow(df['Prod4']))
df.to_csv(processed_data_file, index=False)
