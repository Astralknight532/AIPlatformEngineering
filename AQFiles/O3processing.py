# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:31:17 2020

@author: hanan
"""

# Import needed libraries
import customfunctions as cf # a Python file with functions I wrote
import pandas as pd
import numpy as np
import math as m
#import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
#from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Read in the data set
airpol_data = pd.read_csv(
    "C:/Users/hanan/Desktop/PersonalRepository/AQFiles/pollution_us_2000_2016.csv",
    header = 0, 
    parse_dates = ['Date_Local'],
    infer_datetime_format = True,
    index_col = 0,
    squeeze = True,
    usecols = ['Index', 'Date_Local', 'O3_Mean'],
    encoding = 'utf-8-sig', 
    low_memory = False
)

# Get info about the data set
#print(airpol_data.info())
#print("The 1st 5 rows of the dataset: \n%s\n" % airpol_data.head())
#print("The last 5 rows of the dataset: \n%s" % airpol_data.tail())

# Select the columns for O3 data
# O3 Concentration is in parts per million, Date_Local is in the format YYYY-MM-DD
o3avg = airpol_data[['Date_Local', 'O3_Mean']]

# Handle duplicate values in the data
o3avg = o3avg.drop_duplicates('Date_Local')

# Some of the data (upon analysis) is stored as a string, so it must be converted to a usable form
o3avg['Date_Local'] = cf.dt_convert(o3avg['Date_Local'])
o3avg['O3_Mean'] = cf.float_convert(o3avg['O3_Mean'])

# Handle null values in the data
for c_o3 in o3avg['O3_Mean'].values:
    o3avg['O3_Mean'] = o3avg['O3_Mean'].fillna(o3avg['O3_Mean'].mean())
    
'''
# O3 daily avg. concentration (in PPM)
o3fig = px.scatter(o3avg, x = 'Date_Local', y = 'O3_Mean', width = 3000, height = 2500)
o3fig.add_trace(go.Scatter(
    x = o3avg['Date_Local'],
    y = o3avg['O3_Mean'],
    name = 'O3',
    line_color = 'green',
    opacity = 0.8  
))
o3fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. O3 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per million)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
o3fig.update_xaxes(automargin = True)
o3fig.update_yaxes(automargin = True)
o3fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_o3.png')

# Split the data into train/test sets based on the date
o3mask_train = (o3avg['Date_Local'] < '2010-01-01')
o3mask_test = (o3avg['Date_Local'] >= '2010-01-01')
o3train, o3test = o3avg.loc[o3mask_train], o3avg.loc[o3mask_test]

#print(o3train.info("O3 training set info: \n%s\n" % o3train.info()))
#print(o3test.info("O3 testing set info: \n%s\n" % o3train.info()))

# Univariate forecast setup
TRAIN_SPLIT = 3653
tf.random.set_seed(15)
def uni_dt(ds, start_i, end_i, histsize, tgtsize):
    data = []
    labels = []
    start_i = start_i + histsize
    if end_i is None:
        end_i = len(ds) - tgtsize
        
    for i in range(start_i, end_i):
        ind = range(i - histsize, i)
        data.append(np.reshape(ds[ind], (histsize, 1)))
        labels.append(ds[i + tgtsize])
        
    return np.array(data), np.array(labels)
    
# O3 setup
o3uni = o3avg['O3_Mean']
o3uni.index = o3avg['Date_Local']
o3uni = o3uni.values
o3uni_mean = o3uni[:TRAIN_SPLIT].mean()
o3uni_std = o3uni[:TRAIN_SPLIT].std()
o3uni = (o3uni - o3uni_mean)/o3uni_std 
#print(o3uni.head())
#o3uni.plot(subplots = True)
'''