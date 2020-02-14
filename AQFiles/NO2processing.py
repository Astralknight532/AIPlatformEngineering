# -*- coding: utf-8 -*-

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
    usecols = ['Index', 'Date_Local', 'NO2_Mean'],
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

# Handling duplicate rows in each dataframe
no2avg = no2avg.drop_duplicates('Date_Local') # NO2 dataframe

# Converting the Date_Local column to a datetime object
no2avg['Date_Local'] = cf.dt_convert(no2avg['Date_Local'])

# Some of the data to be analyzed are stored as strings instead of numbers, so
# conversion is needed (the approach I've chosen requires regex)
# Daily average concentrations of each pollutant
no2avg['NO2_Mean'] = cf.float_convert(no2avg['NO2_Mean'])

# Since there are null values in the data, I've chosen to fill them with the mean
# Null values in the Mean Concentration column in the NO2 dataframe
for c_no2 in no2avg['NO2_Mean'].values:
    no2avg['NO2_Mean'] = no2avg['NO2_Mean'].fillna(no2avg['NO2_Mean'].mean())

#print(no2avg.head())
#print(no2avg.info())

from statsmodels.tsa.arima_model import ARIMA
# Univariate forecast setup
# NO2 setup
no2uni = no2avg['NO2_Mean']
no2uni.index = no2avg['Date_Local']
no2uni = no2uni.values
no2size = int(len(no2uni) * 0.66)
no2train, no2test = no2uni[0:no2size], no2uni[no2size:len(no2uni)]
no2hist = [x for x in no2train]
pred = list()
for t in range(len(no2test)):
    no2mod = ARIMA(no2hist, order = (5,1,1))
    no2modfit = no2mod.fit(disp = 0)
    no2out = no2modfit.forecast()
    no2_yhat = no2out[0]
    pred.append(no2_yhat)
    obs = no2test[t]
    no2hist.append(obs)
    #print('Predicted = %f, Expected = %f' % (no2_yhat, obs))

no2error = m.sqrt(mean_squared_error(no2test, pred))
print('NO2 Test RMSE: %.3f' % no2error)
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('US Daily Avg. NO2 Concentration')
plt.xlabel('Time')
plt.ylabel('Daily Avg. NO2 Conc. (PPB)')
plt.plot(no2test, label = 'Test')
plt.plot(pred, color = 'red', label = 'Predict')
plt.legend()
plt.show()

''' 
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
'''

'''
# Splitting the data into train & test sets based on the date
# NO2 sets
no2mask_train = (no2avg['Date_Local'] < '2010-01-01')
no2mask_test = (no2avg['Date_Local'] >= '2010-01-01')
no2train, no2test = no2avg.loc[no2mask_train], no2avg.loc[no2mask_test]

#print("NO2 training set info: \n%s\n" % no2train.info()) #3653 train, 366 test
#print("NO2 testing set info: \n%s\n" % no2test.info())

# Trying out the Keras TimeSeriesGenerator functionality with a LSTM model
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

# Defining a model
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
print('Predicted daily avg. NO2 concentration: %s parts per billion' % yhat[0][0])
print(no2avg['NO2_Mean'].tail())
'''