# -*- coding: utf-8 -*-

# Import needed libraries
import customfunctions as cf # a Python file with functions I wrote
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD
from keras.preprocessing.sequence import TimeseriesGenerator
from numpy import array
#import matplotlib.pyplot as plt
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
    
# Splitting the data into train & test sets based on the date
comask_train = (co_avg['Date_Local'] < '2010-01-01')
comask_test = (co_avg['Date_Local'] >= '2010-01-01')
co_train, co_test = co_avg.loc[comask_train], co_avg.loc[comask_test]

#print("O3 training set info: \n%s\n" % o3train.info()) #3653 train, 366 test
#print("O3 testing set info: \n%s\n" % o3test.info())

# Using the Keras TimeSeriesGenerator functionality to build a LSTM model
ser = array(co_avg['CO_Mean'].values)
n_feat = 1
ser = ser.reshape((len(ser), n_feat))
n_in = 2
tsg = TimeseriesGenerator(ser, ser, length = n_in, batch_size = 20)
print('Number of samples: %d' % len(tsg))

# Defining an alternative optimizer
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

# Defining a model
co_mod = Sequential([
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
co_mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = co_mod.fit_generator(
    tsg, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Save the model in a HDF5 file format (as a .h5 file)
path = 'C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/co_model.h5'
co_mod.save(path, overwrite = True)

# Test prediction
x_in = array(co_test['CO_Mean'].tail(2)).reshape((1, n_in, n_feat))
co_pred = co_mod.predict(x_in, verbose = 0)
print('Predicted daily avg. CO concentration: %.3f parts per million' % co_pred[0][0])
print(co_avg['CO_Mean'].tail())

'''
# Plotting the metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('CO Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.show()
'''

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
