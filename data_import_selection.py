""""
Run to import the data as raw datafiles and select the specific datapoints used in the analysis period and analysis location. 

Performed offline as dataset is large.
""""

import pandas as pd

with open('hour_data.txt') as f:
    for i, line in enumerate(f):
        if line.startswith('# STN'):
            header_row = i
            break
          
# Weather dataframe
knmi = pd.read_csv(
    'hour_data.txt',
    skiprows=header_row,
    skipinitialspace=True
)

knmi_selected = knmi[(knmi['YYYYMMDD'] >= 20260101) & (knmi['YYYYMMDD'] <= 20260114)]
knmi_selected = knmi_selected.reset_index(drop = True)
knmi_selected.to_csv('weather_selected.csv', index=False)

# Train dataframe
df = = pd.read_csv('services-2026-01.csv')

df_utrecht = df[df['Stop:Station name'] == 'Utrecht Centraal']
df_utrecht=df_utrecht.reset_index()
df_utrecht
df_utrecht.to_csv('traindata_selected.csv', index=False)
