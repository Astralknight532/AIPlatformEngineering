# -*- coding: utf-8 -*-

# Import needed libraries
import customfunctions as cf # a Python file with functions I wrote
import pandas as pd
import math as m
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD
from keras.preprocessing.sequence import TimeseriesGenerator
from numpy import array
#import plotly.graph_objects as go
#import plotly.express as px
#import os

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
    
# Splitting the data into train & test sets based on the date
o3mask_train = (o3avg['Date_Local'] < '2010-01-01')
o3mask_test = (o3avg['Date_Local'] >= '2010-01-01')
o3train, o3test = o3avg.loc[o3mask_train], o3avg.loc[o3mask_test]

#print("O3 training set info: \n%s\n" % o3train.info()) #3653 train, 366 test
#print("O3 testing set info: \n%s\n" % o3test.info())

# Using the Keras TimeSeriesGenerator functionality to build a LSTM model
ser = array(o3avg['O3_Mean'].values)
n_feat = 1
ser = ser.reshape((len(ser), n_feat))
n_in = 2
tsg = TimeseriesGenerator(ser, ser, length = n_in, batch_size = 20)
print('Number of samples: %d' % len(tsg))

# Defining an alternative optimizer
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

# Defining a model
o3mod = Sequential([
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

# Compiling & fitting the model
o3mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = o3mod.fit_generator(
    tsg, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Test prediction
x_in = array(o3test['O3_Mean'].tail(2)).reshape((1, n_in, n_feat))
o3pred = o3mod.predict(x_in, verbose = 0)
print('Predicted daily avg. O3 concentration: %.3f parts per million' % o3pred[0][0])
#print(o3avg['O3_Mean'].tail())

# Plotting the metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
for i in range(len(history.history['mse'])):
    history.history['mse'][i] = m.sqrt(history.history['mse'][i])
    
plt.plot(history.history['mse'], label = 'RMSE', color = 'green')
plt.legend()
plt.show()

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
'''