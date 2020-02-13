# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import needed libraries
import customfunctions as cf # a Python file with functions I wrote
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD
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
    usecols = ['Index', 'Date_Local', 'NO2_Mean', 'O3_Mean', 'SO2_Mean', 'CO_Mean'],
    encoding = 'utf-8-sig', 
    low_memory = False
)

# Get info about the data set
#print(airpol_data.info())
#print("The 1st 5 rows of the dataset: \n%s\n" % airpol_data.head())
#print("The last 5 rows of the dataset: \n%s" % airpol_data.tail())

# Handling the data so that it can be used for ML/DL 
# Separate the data by type of pollutant
# Note: Date_Local is written as YYYY-MM-DD
# These 2 chemicals' concentrations are recorded in PPB (parts per billion)
no2avg = airpol_data[['Date_Local', 'NO2_Mean']]
so2avg = airpol_data[['Date_Local', 'SO2_Mean']]
# These 2 chemicals' concentrations are recorded in PPM (parts per million)
o3avg = airpol_data[['Date_Local', 'O3_Mean']]
co_avg = airpol_data[['Date_Local', 'CO_Mean']]

# Handling duplicate rows in each dataframe
no2avg = no2avg.drop_duplicates('Date_Local') # NO2 dataframe
so2avg = so2avg.drop_duplicates('Date_Local') # SO2 dataframe
o3avg = o3avg.drop_duplicates('Date_Local') # O3 dataframe
co_avg = co_avg.drop_duplicates('Date_Local') # CO dataframe

# Converting the Date_Local column to a datetime object
no2avg['Date_Local'] = cf.dt_convert(no2avg['Date_Local'])
so2avg['Date_Local'] = cf.dt_convert(so2avg['Date_Local'])
o3avg['Date_Local'] = cf.dt_convert(o3avg['Date_Local'])
co_avg['Date_Local'] = cf.dt_convert(co_avg['Date_Local'])

# Some of the data to be analyzed are stored as strings instead of numbers, so
# conversion is needed (the approach I've chosen requires regex)
# Daily average concentrations of each pollutant
no2avg['NO2_Mean'] = cf.float_convert(no2avg['NO2_Mean'])
so2avg['SO2_Mean'] = cf.float_convert(so2avg['SO2_Mean']) 
so2avg['SO2_Mean'].fillna(3, inplace = True, limit = 1) # Handling one of the missing values at the beginning of the data
o3avg['O3_Mean'] = cf.float_convert(o3avg['O3_Mean'])
co_avg['CO_Mean'] = cf.float_convert(co_avg['CO_Mean'])

# Since there are null values in the data, I've chosen to fill them with the mean
# Null values in the Mean Concentration column in the NO2 dataframe
for c_no2 in no2avg['NO2_Mean'].values:
    no2avg['NO2_Mean'] = no2avg['NO2_Mean'].fillna(no2avg['NO2_Mean'].mean())

# Null values in the Mean Concentration column in the SO2 dataframe
for c_so2 in so2avg['SO2_Mean'].values:
    so2avg['SO2_Mean'] = so2avg['SO2_Mean'].fillna(so2avg['SO2_Mean'].mean())

# Null values in the Mean Concentration column in the O3 dataframe
for c_o3 in o3avg['O3_Mean'].values:
    o3avg['O3_Mean'] = o3avg['O3_Mean'].fillna(o3avg['O3_Mean'].mean())

# Null values in the Mean Concentration column in the CO dataframe
for c_co in co_avg['CO_Mean'].values:
    co_avg['CO_Mean'] = co_avg['CO_Mean'].fillna(co_avg['CO_Mean'].mean())

#print(no2avg.head(), so2avg.head(), o3avg.head(), co_avg.head())
#print(no2avg.info(), so2avg.info(), o3avg.info(), co_avg.info())
    
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
        
# NO2 setup
no2uni = no2avg['NO2_Mean']
no2uni.index = no2avg['Date_Local']
no2uni = no2uni.values
no2uni_mean = no2uni[:TRAIN_SPLIT].mean()
no2uni_std = no2uni[:TRAIN_SPLIT].std()
no2uni = (no2uni - no2uni_mean)/no2uni_std 
#print(no2uni.head())
#no2uni.plot(subplots = True)

# SO2 setup
so2uni = so2avg['SO2_Mean']
so2uni.index = so2avg['Date_Local']
so2uni = so2uni.values
so2uni_mean = so2uni[:TRAIN_SPLIT].mean()
so2uni_std = so2uni[:TRAIN_SPLIT].std()
so2uni = (so2uni - so2uni_mean)/so2uni_std 
#print(so2uni.head())
#so2uni.plot(subplots = True)

# O3 setup
o3uni = o3avg['O3_Mean']
o3uni.index = o3avg['Date_Local']
o3uni = o3uni.values
o3uni_mean = o3uni[:TRAIN_SPLIT].mean()
o3uni_std = o3uni[:TRAIN_SPLIT].std()
o3uni = (o3uni - o3uni_mean)/o3uni_std 
#print(o3uni.head())
#o3uni.plot(subplots = True)

# CO setup
co_uni = co_avg['CO_Mean']
co_uni.index = co_avg['Date_Local']
co_uni = co_uni.values
co_uni_mean = co_uni[:TRAIN_SPLIT].mean()
co_uni_std = co_uni[:TRAIN_SPLIT].std()
co_uni = (co_uni - co_uni_mean)/co_uni_std 
#print(co_uni.head())
#co_uni.plot(subplots = True)

# Creating the models
# NO2 model
no2_hist = 20
no2_tgt = 0
no2_xtrain, no2_ytrain = uni_dt(no2uni, 0, TRAIN_SPLIT, no2_hist, no2_tgt)
no2_xvaluni, no2_yvaluni = uni_dt(no2uni, TRAIN_SPLIT, None, no2_hist, no2_tgt)
#print('1 window of past history: \n%s\n' % no2_xtrain[0])
#print('Target NO2 concentration to predict: \n%s' % no2_ytrain[0])
def create_ts(l):
    return list(range(-l, 0))

def showplot(plotdata, d, t):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    ts = create_ts(plotdata[0].shape[0])
    if d:
        future = d
    else:
        future = 0
    
    plt.title(t)
    for i, x in enumerate(plotdata):
        if i:
            plt.plot(future, plotdata[i], marker[i], markersize = 10, label = labels[i])
        else:
            plt.plot(ts, plotdata[i].flatten(), marker[i], label = labels[i])
    plt.legend()
    plt.xlim([ts[0], (future + 5) * 2])
    plt.xlabel('Time_Step')
    return plt

#showplot([no2_xtrain[0], no2_ytrain[0]], 0, 'Sample Example')

def baseline(h):
    return np.mean(h)

#showplot([no2_xtrain[0], no2_ytrain[0], baseline(no2_xtrain[0])], 0, 'Baseline Prediction Example')

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_no2uni = tf.data.Dataset.from_tensor_slices((no2_xtrain, no2_ytrain))
train_no2uni = train_no2uni.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_no2uni = tf.data.Dataset.from_tensor_slices((no2_xvaluni, no2_yvaluni))
val_no2uni = val_no2uni.batch(BATCH_SIZE).repeat()

# Define the structure of the model
mp = Sequential([
    LSTM(50, activation = 'relu', input_shape = no2_xtrain.shape[-2:], return_sequences = True),
    Dropout(0.2),
    LSTM(50, return_sequences = True),
    Dropout(0.2),
    LSTM(50, return_sequences = True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

EVAL_INT = 10
EPOCHS = 500
opt = SGD( #alternative optimizer to try
    lr = 0.01, 
    momentum = 0.9, 
    nesterov = True
) 
mp.compile(
    optimizer = 'adam', 
    loss = 'mean_squared_logarithmic_error', 
    metrics = ['mse']
)
mp.fit(
    train_no2uni, 
    steps_per_epoch = EVAL_INT, 
    epochs = EPOCHS, 
    verbose = 0, 
    validation_data = val_no2uni, 
    validation_steps = 50
)

for x, y in val_no2uni.take(3):
    plot = showplot([x[0].numpy(), y[0].numpy(), mp.predict(x)[0]], 0, 'LSTM Model')
    plot.show()

'''       
# Splitting the data into train & test sets based on the date
# NO2 sets
no2mask_train = (no2avg['Date_Local'] < '2010-01-01')
no2mask_test = (no2avg['Date_Local'] >= '2010-01-01')
no2train, no2test = no2avg.loc[no2mask_train], no2avg.loc[no2mask_test]
# SO2 sets
so2mask_train = (so2avg['Date_Local'] < '2010-01-01')
so2mask_test = (so2avg['Date_Local'] >= '2010-01-01')
so2train, so2test = so2avg.loc[so2mask_train], so2avg.loc[so2mask_test]
# O3 sets
o3mask_train = (o3avg['Date_Local'] < '2010-01-01')
o3mask_test = (o3avg['Date_Local'] >= '2010-01-01')
o3train, o3test = o3avg.loc[o3mask_train], o3avg.loc[o3mask_test]
# CO sets
co_mask_train = (co_avg['Date_Local'] < '2010-01-01')
co_mask_test = (co_avg['Date_Local'] >= '2010-01-01')
co_train, co_test = co_avg.loc[co_mask_train], co_avg.loc[co_mask_test]

#print("NO2 training set info: \n%s\n" % no2train.info()) #3653 train, 366 test
#print("NO2 testing set info: \n%s\n" % no2test.info())
#print(so2train.info("SO2 training set info: \n%s\n" % so2train.info()))
#print(so2test.info("SO2 testing set info: \n%s\n" % so2test.info()))
#print(o3train.info("O3 training set info: \n%s\n" % o3train.info()))
#print(o3test.info("O3 testing set info: \n%s\n" % o3train.info()))
#print(co_train.info("CO training set info: \n%s\n" % co_train.info()))
#print(co_test.info("CO testing set info: \n%s" % co_train.info()))
'''

'''
# Plotting the daily average concentration of each pollutant
# Checking for the folder that figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures'):
    os.mkdir('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures')
    
# NO2 daily avg. concentration (in PPB)
no2fig = px.scatter(no2avg, x = 'Date_Local', y = 'NO2_Mean', width = 3000, height = 2500)
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
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per billion)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
no2fig.update_xaxes(automargin = True)
no2fig.update_yaxes(automargin = True)
no2fig.write_image('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures/avg_no2.png')

# SO2 daily avg. concentration (in PPB)
so2fig = px.scatter(so2avg, x = 'Date_Local', y = 'SO2_Mean', width = 3000, height = 2500)
so2fig.add_trace(go.Scatter(
    x = so2avg['Date_Local'],
    y = so2avg['SO2_Mean'],
    name = 'SO2',
    line_color = 'blue',
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
'''
# Trying out the Keras TimeSeriesGenerator functionality
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator

ser = array(no2avg['NO2_Mean'].values)
n_feat = 1
ser = ser.reshape((len(ser), n_feat))
n_in = 2
tsg = TimeseriesGenerator(ser, ser, length = n_in, batch_size = 20)
print('Number of samples: %d' % len(tsg))

# Defining an alternative optimizer
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

# Defining a simple Multilayer Perceptron model
mp = Sequential([
    LSTM(50, activation = 'relu', input_shape = (n_in, n_feat), return_sequences = True),
    Dropout(0.2),
    LSTM(50, return_sequences = True),
    Dropout(0.2),
    LSTM(50, return_sequences = True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
mp.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = mp.fit_generator(tsg, steps_per_epoch = 10, epochs = 500, verbose = 0)

# Plotting the training loss
plt.rcParams['figure.figsize'] = (20, 10)
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label = 'train')
#plt.plot(history.history['val_loss'], label = 'test')
plt.legend()
# Plotting the training MSE (mean square error)
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history.history['mse'], label = 'train')
#plt.plot(history.history['val_mean_squared_error'], label = 'test')
plt.legend()
# Plotting the actual values vs. the predicted values

# Test prediction
x_in = array(no2test['NO2_Mean'].tail(2)).reshape((1, n_in, n_feat))
yhat = mp.predict(x_in, verbose = 0)
print(yhat)
print(no2avg['NO2_Mean'].tail())
'''
