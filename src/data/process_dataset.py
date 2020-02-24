import numpy as np
import pandas as pd

def gaussian_white_noise(column):
    length = len(column)
    mu, sigma = [column.mean(), column.std()]
    noise = np.random.normal(loc=mu, scale=sigma, size=length)
    for i in range(length):
        column[i] += noise[i]
    return column

def net_flow(production):
    net = []
    for prod in production:
        if len(net) == 0:
            net.append(prod)
        else:
            net.append(net[-1] + prod)
    return net

input_filename = "~/ut/research/crm_validation/data/raw/CRMP_Corrected_July16_2018.xlsx"
output_filename = "~/ut/research/crm_validation/data/interim/CRMP_Corrected_July16_2018.csv"

df = pd.read_excel(input_filename, 0, skiprows=9)
df = df.drop(df.columns[[3, 4, 7, 8]], axis=1)
df.columns = [
    'Time', 'Random_inj1', 'Random_inj2', 'Fixed_inj1', 'Fixed_inj2',
    'Prod1', 'Prod2', 'Prod3', 'Prod4'
]
columns = df.columns[3:]
for column in columns:
    df[column] = gaussian_white_noise(df[column])

df.insert(4, 'Net_Fixed_inj1', net_flow(df['Fixed_inj1']))
df.insert(6, 'Net_Fixed_inj2', net_flow(df['Fixed_inj2']))
df.insert(8, 'Net_Prod1', net_flow(df['Prod1']))
df.insert(10, 'Net_Prod2', net_flow(df['Prod2']))
df.insert(12, 'Net_Prod3', net_flow(df['Prod3']))
df.insert(14, 'Net_Prod4', net_flow(df['Prod4']))
df.to_csv(output_filename, index=False)
