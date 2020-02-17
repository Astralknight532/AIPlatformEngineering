# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:31:23 2020

@author: hanan
"""

# Import needed libraries
import customfunctions as cf # a Python file with functions I wrote
import pandas as pd
#import math as m
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#import plotly.express as px
#import os
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
#from keras.optimizers import SGD


# Read in the data set
airpol_data = pd.read_csv(
    "C:/Users/hanan/Desktop/PersonalRepository/AQFiles/pollution_us_2000_2016.csv",
    header = 0, 
    parse_dates = ['Date_Local'],
    infer_datetime_format = True,
    index_col = 0,
    squeeze = True,
    usecols = ['Index', 'Date_Local', 'CO_Mean'],
    encoding = 'utf-8-sig', 
    low_memory = False
)

# Get info about the data set
#print(airpol_data.info())
#print("The 1st 5 rows of the dataset: \n%s\n" % airpol_data.head())
#print("The last 5 rows of the dataset: \n%s" % airpol_data.tail())

# Selecting the columns for CO data
# CO concentration is in parts per million, Date_Local is in the format YYYY-MM-DD
co_avg = airpol_data[['Date_Local', 'CO_Mean']]

# Handle duplicate values in the data
co_avg = co_avg.drop_duplicates('Date_Local') # CO dataframe

# Some of the data (upon analysis) is stored as a string, so it must be converted to a usable form
co_avg['Date_Local'] = cf.dt_convert(co_avg['Date_Local'])
co_avg['CO_Mean'] = cf.float_convert(co_avg['CO_Mean'])

# Handle null values in the data
for c_co in co_avg['CO_Mean'].values:
    co_avg['CO_Mean'] = co_avg['CO_Mean'].fillna(co_avg['CO_Mean'].mean())
    
'''
# CO daily avg. concentration (in PPM)
co_fig = px.scatter(co_avg, x = 'Date_Local', y = 'CO_Mean', width = 3000, height = 2500)
co_fig.add_trace(go.Scatter(
    x = co_avg['Date_Local'],
    y = co_avg['CO_Mean'],
    name = 'CO',
    line_color = 'yellow',
    opacity = 0.8    
))
co_fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. CO Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per million)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
co_fig.update_xaxes(automargin = True)
co_fig.update_yaxes(automargin = True)
co_fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_co.png')
'''
