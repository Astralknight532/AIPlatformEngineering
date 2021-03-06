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
import os

# Read in the data set
airpol_data = pd.read_csv(
    "C:/Users/hanan/Desktop/Personal Projects/AIPlatformEngineering/AQFiles/pollution_us_2000_2016.csv",
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

# Selecting the columns of NO2 data
# NO2 concentration is in parts per billion,  Date_Local is in the format YYYY-MM-DD
no2avg = airpol_data[['Date_Local', 'NO2_Mean']]

# Handling duplicate rows in each dataframe
no2avg = no2avg.drop_duplicates('Date_Local') # NO2 dataframe

# Some of the data is stored as strings instead of numbers, so conversion is needed
no2avg['Date_Local'] = cf.dt_convert(no2avg['Date_Local'])
no2avg['NO2_Mean'] = cf.float_convert(no2avg['NO2_Mean'])

# Handle null values in the data
for c_no2 in no2avg['NO2_Mean'].values:
    no2avg['NO2_Mean'] = no2avg['NO2_Mean'].fillna(no2avg['NO2_Mean'].mean())

# Sort the data by the date from earliest to latest
no2avg.sort_values(by = ['Date_Local'], ascending = True, inplace = True, kind = 'mergesort')
#print(no2avg.head())
#print(no2avg.tail())

# Checking for the folder that cleaned data will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/cleanData'):
    os.mkdir('C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/cleanData')
    
# Write the cleaned data to a separate CSV file
cleaned_no2csv = "C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/cleanData/cleaned_NO2data.csv"
no2avg.to_csv(cleaned_no2csv, date_format = '%Y-%m-%d')   

# Splitting the data into train & test sets based on the date
no2mask_train = (no2avg['Date_Local'] < '2010-01-01')
no2mask_test = (no2avg['Date_Local'] >= '2010-01-01')
no2train, no2test = no2avg.loc[no2mask_train], no2avg.loc[no2mask_test]

#print("NO2 training set info: \n%s\n" % no2train.info()) # 3653 train, 366 test
#print("NO2 testing set info: \n%s\n" % no2test.info())

# Using the Keras TimeSeriesGenerator functionality to build a LSTM model
ser_train = array(no2train['NO2_Mean'].values)
ser_test = array(no2test['NO2_Mean'].values)
n_feat = 1
ser_train = ser_train.reshape((len(ser_train), n_feat))
n_in = 2
train_gen = TimeseriesGenerator(ser_train, ser_train, length = n_in, sampling_rate = 1, batch_size = 10)
test_gen = TimeseriesGenerator(ser_test, ser_test, length = n_in, sampling_rate = 1, batch_size = 1)
#print('Number of training samples: %d' % len(train_gen))
#print('Number of testing samples: %d' % len(test_gen))

# Defining an alternative optimizer (instead of ADAM optimizer)
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

# Defining a model
no2mod = Sequential([
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
no2mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = no2mod.fit_generator(
    train_gen, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Getting a summary of the model
#print(no2mod.summary())

# Save the model in a HDF5 file format (as a .h5 file)
path = 'C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/SavedModels/no2_model.h5'
no2mod.save(path, overwrite = True)

# Test prediction
x_in = array(no2test['NO2_Mean'].head(n_in)).reshape((1, n_in, n_feat))
no2pred = no2mod.predict(x_in, verbose = 0)
print('Predicted daily avg. NO2 concentration: %.3f parts per billion\n' % no2pred[0][0])
print(no2avg[no2avg['Date_Local'] == '2010-01-03'])

'''
# Plotting the metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('NO2 Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.show()

# Plotting the daily average concentration of each pollutant
# Checking for the folder that figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/plotlyfigures'):
    os.mkdir('C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/plotlyfigures')
    
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
no2fig.write_image('C:/Users/hanan/Desktop/Personal Project/AIPlatformEngineering/AQFiles/plotlyfigures/avg_no2.png')
'''