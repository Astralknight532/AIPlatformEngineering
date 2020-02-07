# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import needed libraries
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
#import math as m
#import matplotlib.pyplot as plt

# Read in the data set
airpol_data = pd.read_csv("C:/Users/hanan/Desktop/PersonalRepository/AQFiles/pollution_us_2000_2016.csv", encoding = 'utf-8-sig', low_memory = False)

# Get info about the data set
#print(airpol_data.info())
#print("The 1st 5 rows of the dataset: \n%s\n" % airpol_data.head())
#print("The last 5 rows of the dataset: \n%s" % airpol_data.tail())

# Handling the data so that it can be used for ML/DL 
# Separate the data by type of pollutant
# Note: Date_Local is written as YYYY-MM-DD & 1stMaxHour uses the 24-hour time system 
# These 2 chemicals' concentrations are recorded in PPB (parts per billion)
no2avg = airpol_data[['Date_Local', 'NO2_Mean']]
no2max = airpol_data[['Date_Local', 'NO2_1stMaxValue', 'NO2_1stMaxHour']]
so2avg = airpol_data[['Date_Local', 'SO2_Mean']]
so2max = airpol_data[['Date_Local', 'SO2_1stMaxValue', 'SO2_1stMaxHour']]
# These 2 chemicals' concentrations are recorded in PPM (parts per million)
o3avg = airpol_data[['Date_Local', 'O3_Mean']]
o3max = airpol_data[['Date_Local', 'O3_1stMaxValue', 'O3_1stMaxHour']]
co_avg = airpol_data[['Date_Local', 'CO_Mean']]
co_max = airpol_data[['Date_Local', 'CO_1stMaxValue', 'CO_1stMaxHour']]

# Handling duplicate rows in each dataframe
# NO2 dataframes
no2avg = no2avg.drop_duplicates('Date_Local')
no2max = no2max.drop_duplicates('Date_Local')
# SO2 dataframes
so2avg = so2avg.drop_duplicates(['Date_Local', 'SO2_Mean'])
so2max = so2max.drop_duplicates(['Date_Local', 'SO2_1stMaxValue'])
# O3 dataframes
o3avg = o3avg.drop_duplicates('Date_Local')
o3max = o3max.drop_duplicates('Date_Local')
# CO dataframes
co_avg = co_avg.drop_duplicates(['Date_Local', 'CO_Mean'])
co_max = co_max.drop_duplicates(['Date_Local', 'CO_1stMaxValue'])

# Defining 2 functions to organize repetitive tasks
# Converts date column to datetime
def dt_convert(data:pd.Series) -> pd.Series: 
    data = pd.to_datetime(data.str.strip(), format = '%Y-%m-%d', errors = 'coerce')
    return data
# Converts the passed string values into type float64 - extracts strings using regex
def float_convert(data:pd.Series) -> pd.Series:
    data = data.str.extract('(\d*\.\d*)').astype('float64')
    return data

# Converting the Date_Local column to a datetime object and setting it as the index
no2avg['Date_Local'] = pd.to_datetime(no2avg['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
no2max['Date_Local'] = pd.to_datetime(no2max['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
so2avg['Date_Local'] = pd.to_datetime(so2avg['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
so2max['Date_Local'] = pd.to_datetime(so2max['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
o3avg['Date_Local'] = pd.to_datetime(o3avg['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
o3max['Date_Local'] = pd.to_datetime(o3max['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
co_avg['Date_Local'] = pd.to_datetime(co_avg['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')
co_max['Date_Local'] = pd.to_datetime(co_max['Date_Local'].str.strip(), format = '%Y-%m-%d', errors = 'coerce')

# Some of the data to be analyzed are stored as strings instead of numbers, so
# conversion is needed (the approach I've chosen requires regex)

# Daily average concentrations of each pollutant
no2avg['NO2_Mean'] = no2avg['NO2_Mean'].str.extract('(\d*\.\d*)').astype('float64')
#print(no2avg['NO2_Mean'].head())

so2avg['SO2_Mean'] = so2avg['SO2_Mean'].str.extract('(\d*\.\d*)').astype('float64') 
so2avg['SO2_Mean'].fillna(3, inplace = True, limit = 1)
#print(so2avg['SO2_Mean'].head())

o3avg['O3_Mean'] = o3avg['O3_Mean'].str.extract('(\d*\.\d*)').astype('float64')
#print(o3avg['O3_Mean'].head())

co_avg['CO_Mean'] = co_avg['CO_Mean'].str.extract('(\d*\.\d*)').astype('float64')
#print(co_avg['CO_Mean'].head())

# Daily max concentrations of each pollutant

# Plotting the daily average concentration of each pollutant
# Checking for the folder that figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures'):
    os.mkdir('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures')
    
# NO2 daily avg. concentration (in PPB)
no2fig = px.scatter(no2avg, x = 'Date_Local', y = 'NO2_Mean', width = 1500, height = 1000)
no2fig.add_trace(go.Scatter(
    x = no2avg['Date_Local'],
    y = no2avg['NO2_Mean'],
    name = 'NO2',
    line_color = 'red',
    opacity = 0.8    
))
no2fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. NO2 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per billion)')
)
no2fig.update_xaxes(automargin = True)
no2fig.update_yaxes(automargin = True)
no2fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_no2.png')

# SO2 daily avg. concentration (in PPB)
so2fig = px.scatter(no2avg, x = 'Date_Local', y = 'SO2_Mean', width = 1500, height = 1000)
so2fig.add_trace(go.Scatter(
    x = so2avg['Date_Local'],
    y = so2avg['SO2_Mean'],
    name = 'SO2',
    line_color = 'red',
    opacity = 0.8    
))
so2fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. SO2 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per billion)')
)
so2fig.update_xaxes(automargin = True)
so2fig.update_yaxes(automargin = True)
so2fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_so2.png')

# O3 daily avg. concentration (in PPM)
o3fig = px.scatter(no2avg, x = 'Date_Local', y = 'O3_Mean', width = 1500, height = 1000)
o3fig.add_trace(go.Scatter(
    x = o3avg['Date_Local'],
    y = o3avg['O3_Mean'],
    name = 'O3',
    line_color = 'red',
    opacity = 0.8    
))
o3fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. O3 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per million)')
)
o3fig.update_xaxes(automargin = True)
o3fig.update_yaxes(automargin = True)
o3fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_o3.png')

# CO daily avg. concentration (in PPM)
co_fig = px.scatter(no2avg, x = 'Date_Local', y = 'CO_Mean', width = 1500, height = 1000)
co_fig.add_trace(go.Scatter(
    x = co_avg['Date_Local'],
    y = co_avg['CO_Mean'],
    name = 'CO',
    line_color = 'red',
    opacity = 0.8    
))
co_fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. CO Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per million)')
)
co_fig.update_xaxes(automargin = True)
co_fig.update_yaxes(automargin = True)
co_fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_co.png')