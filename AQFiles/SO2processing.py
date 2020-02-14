# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:30:46 2020

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
    usecols = ['Index', 'Date_Local', 'SO2_Mean'],
    encoding = 'utf-8-sig', 
    low_memory = False
)

# Get info about the data set
#print(airpol_data.info())
#print("The 1st 5 rows of the dataset: \n%s\n" % airpol_data.head())
#print("The last 5 rows of the dataset: \n%s" % airpol_data.tail())

# Select the columns for SO2 data
# Concentration for SO2 is in parts per billion, Date_Local is in the format YYYY-MM-YY
so2avg = airpol_data[['Date_Local', 'SO2_Mean']]

# Handle duplicate values in the data
so2avg = so2avg.drop_duplicates('Date_Local')

# Some of the data (upon analysis) is stored as a string, so it must be converted to a usable form
so2avg['Date_Local'] = cf.dt_convert(so2avg['Date_Local'])
so2avg['SO2_Mean'] = cf.float_convert(so2avg['SO2_Mean']) 
so2avg['SO2_Mean'].fillna(3, inplace = True, limit = 1) # Handling one of the missing values at the beginning of the data

# Handle null values in the data
for c_so2 in so2avg['SO2_Mean'].values:
    so2avg['SO2_Mean'] = so2avg['SO2_Mean'].fillna(so2avg['SO2_Mean'].mean())
    
'''
# SO2 daily avg. concentration (in PPB)
so2fig = px.scatter(so2avg, x = 'Date_Local', y = 'SO2_Mean', width = 3000, height = 2500)
so2fig.add_trace(go.Scatter(
    x = so2avg['Date_Local'],
    y = so2avg['SO2_Mean'],
    name = 'SO2',
    line_color = 'black',
    opacity = 0.8  
))
so2fig.update_layout(
    xaxis_range = ['2000-01-01', '2011-12-31'], 
    title_text = 'US Daily Avg. SO2 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per billion)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
so2fig.update_xaxes(automargin = True)
so2fig.update_yaxes(automargin = True)
so2fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_so2.png')

# Splitting the data into train/test sets based on the date
so2mask_train = (so2avg['Date_Local'] < '2010-01-01')
so2mask_test = (so2avg['Date_Local'] >= '2010-01-01')
so2train, so2test = so2avg.loc[so2mask_train], so2avg.loc[so2mask_test]

#print(so2train.info("SO2 training set info: \n%s\n" % so2train.info()))
#print(so2test.info("SO2 testing set info: \n%s\n" % so2test.info()))

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

so2uni = so2avg['SO2_Mean']
so2uni.index = so2avg['Date_Local']
so2uni = so2uni.values
so2uni_mean = so2uni[:TRAIN_SPLIT].mean()
so2uni_std = so2uni[:TRAIN_SPLIT].std()
so2uni = (so2uni - so2uni_mean)/so2uni_std 
#print(so2uni.head())
#so2uni.plot(subplots = True)
'''

