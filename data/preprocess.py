import pandas as pd
import numpy as np
import os

air = pd.read_excel('./Air/AirQualityUCI.xlsx', header=0)
air['Date'] = air['Date'].astype('str')
air['Time'] = air['Time'].astype('str')
air['Date'] = air['Date'] + " " + air['Time']
cols = list(air.columns)
cols.remove('Time')
cols.remove('NMHC(GT)')
air = air[cols]
# data -200
cols.remove('Date')
air_values = air[cols].values
print(air_values)
mean_list = []
for i in range(air_values.shape[1]):
    values = air_values[:, i]
    mean_list.append(values[values > -200].mean())
mean_list = np.array(mean_list)
mean_list = np.expand_dims(mean_list, axis=0)
mean_list = mean_list.repeat(air_values.shape[0], axis=0)
air_values = np.where(air_values > -200, air_values, mean_list)
df_air = pd.DataFrame(data=air_values, columns=[cols])
df_air.insert(loc=0, column='date', value=air['Date'])
df_air.to_csv('./Air/Air.csv', mode='w', header=True, index=False)

light = pd.read_csv('./Light/energydata_complete.csv')
cols = list(light.columns)
cols.remove('rv1')
cols.remove('rv2')
cols.remove('date')
light_values = light[cols].values
mean_list = []
for i in range(light_values.shape[1]):
    values = light_values[:, i]
    mean_list.append(values[values > -200].mean())
mean_list = np.array(mean_list)
mean_list = np.expand_dims(mean_list, axis=0)
mean_list = mean_list.repeat(light_values.shape[0], axis=0)
light_values = np.where(light_values > -200, light_values, mean_list)
df_light = pd.DataFrame(data=light_values, columns=[cols])
df_light.insert(loc=0, column='date', value=light['date'])
print(df_light)
df_light.to_csv('./Light/Light.csv', mode='w', header=True, index=False)

