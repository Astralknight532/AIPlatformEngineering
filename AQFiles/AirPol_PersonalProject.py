# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from livelossplot.keras import PlotLossesCallback
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import plotly.express as px
import customfunctions as cf

# Read in the data set
airpol_data = pd.read_csv(
    "C:/Users/hanan/Desktop/PersonalRepository/AQFiles/pollution_us_2000_2016.csv",
    header = 0, 
    parse_dates = ['Date_Local'],
    infer_datetime_format = True,
    index_col = 0,
    squeeze = True,
    usecols = ['Index', 'Date_Local', 'NO2_Mean', 'NO2_1stMaxValue', 
               'NO2_1stMaxHour', 'O3_Mean', 'O3_1stMaxValue', 'O3_1stMaxHour',
               'SO2_Mean', 'SO2_1stMaxValue', 'SO2_1stMaxHour', 'CO_Mean',
               'CO_1stMaxValue', 'CO_1stMaxHour'
               ],
    encoding = 'utf-8-sig', 
    low_memory = False
)

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

# Converting the Date_Local column to a datetime object
no2avg['Date_Local'] = cf.dt_convert(no2avg['Date_Local'])
no2max['Date_Local'] = cf.dt_convert(no2max['Date_Local'])
so2avg['Date_Local'] = cf.dt_convert(so2avg['Date_Local'])
so2max['Date_Local'] = cf.dt_convert(so2max['Date_Local'])
o3avg['Date_Local'] = cf.dt_convert(o3avg['Date_Local'])
o3max['Date_Local'] = cf.dt_convert(o3max['Date_Local'])
co_avg['Date_Local'] = cf.dt_convert(co_avg['Date_Local'])
co_max['Date_Local'] = cf.dt_convert(co_max['Date_Local'])

# Some of the data to be analyzed are stored as strings instead of numbers, so
# conversion is needed (the approach I've chosen requires regex)

# Daily average concentrations of each pollutant
no2avg['NO2_Mean'] = cf.float_convert(no2avg['NO2_Mean'])
so2avg['SO2_Mean'] = cf.float_convert(so2avg['SO2_Mean']) 
so2avg['SO2_Mean'].fillna(3, inplace = True, limit = 1)
o3avg['O3_Mean'] = cf.float_convert(o3avg['O3_Mean'])
co_avg['CO_Mean'] = cf.float_convert(co_avg['CO_Mean'])

# Daily max concentrations of each pollutant
# NO2_1stMaxHour is already of type int64, but the other 3 need to be converted to int64 type
# All max concentrations for each pollutant are already of type float64
so2max['SO2_1stMaxHour'] = so2max['SO2_1stMaxHour'].astype('int64')
o3max['O3_1stMaxHour'] = o3max['O3_1stMaxHour'].astype('int64')
co_max['CO_1stMaxHour'] = co_max['CO_1stMaxHour'].astype('int64')

'''
# Plotting the daily average concentration of each pollutant
# Checking for the folder that figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures'):
    os.mkdir('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/plotlyfigures')
    
# NO2 daily avg. concentration (in PPB)
no2fig = px.scatter(no2avg, x = 'Date_Local', y = 'NO2_Mean', width = 2000, height = 1000)
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
    line_color = 'blue',
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
    line_color = 'green',
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
    line_color = 'yellow',
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

# Plotting the daily max concentration of each pollutant
# NO2 daily max concentration (in PPB)
# SO2 daily max concentration (in PPB)
# O3 daily max concentration (in PPM)
# CO daily max concentration (in PPM)
'''

def ttb(df, size_test = 24, train_seqlen = 24, fc_len = 12, normalize = False):
    train_df = df[:-size_test]
    xtrain, ytrain = window_splitter(train_df)
    temp = df[:fc_len]
    temp = pd.concat([df, temp], ignore_index = True)
    temp[-fc_len:] = np.nan
    xtest, ytest = window_splitter(temp)
    xtest = xtest[xtrain.shape[0]:]
    ytest = ytest[ytrain.shape[0]:]
    for i in range(fc_len):
        ytest[i, :(11 - i)] = np.nan
    if normalize:
        m = train_df.x.mean()
        sd = train_df.x.std()
        xtrain -= m
        xtrain /= sd
        xtest -= m
        xtest /= sd
    return xtrain, ytrain, xtest, ytest

def window_splitter(train_df, train_seqlen = 24, fc_len = 12):
    i = 0
    x, y = [], []
    while i + train_seqlen + fc_len < len(train_df):
        x.append(train_df.x[i:(i + train_seqlen)].values)
        y.append(train_df.x[(i + train_seqlen):(i + train_seqlen + fc_len)].values)
        i += 1
    x = np.array(x).reshape(-1, train_seqlen, 1)
    y = np.array(y).reshape(-1, fc_len)
    return x, y

series = no2avg.x.values
res = seasonal_decompose(series, model = 'additive', freq = 12, two_sided = False)
#plt.plot(res.trend)
#plt.plot(res.seasonal)
#plt.plot(res.resid)
#plt.plot(res.observed)
#plt.show()

deseason_df = pd.DataFrame({'x':res.trend[12:] + res.resid[12:]})
print(deseason_df.head())

des_xtrain, des_ytrain, des_xtest, des_ytest = ttb(deseason_df)
xtrain, ytrain, xtest, ytest = ttb(no2avg)
print(xtrain.shape, ytrain.shape, xtest.shape, ytrain.shape)

def eval_pred(ypred, ytest) -> pd.DataFrame:
    return pd.DataFrame(abs(ytest - ypred)).mean(skipna = True)
    
def plot_eval(ypred, ytest, n = 12):
    score = eval_pred(ypred, ytest)
    print("Mean absolute error test set: ", score.mean())
    plt.figure(figsize = (6, 4))
    plt.plot(np.arange(1, 13), score)
    plt.xticks(np.arange(1, 13))
    plt.xlabel("horizon [months]", size = 15)
    plt.ylabel("MAE", size = 15)
    plt.title("Scores LSTM on test set")
    plt.show()

    #plt.figure(figsize = (6,4))
    #plt.title("LSTM forecasting - test set window")
    #plt.plot(np.arange(1,13), ypred[n:(n + 1)].reshape(-1, 1),label = "predictions")
    #plt.plot(np.arange(1,13), ytest[n:(n + 1)].reshape(-1, 1),label = "true values")
    #plt.xticks(np.arange(1, 13))
    #plt.xlabel("horizon [months]", size = 15)
    #plt.legend()
    #plt.show()
    
timesteps, features, outputs = xtrain.shape[1], xtrain.shape[2], ytrain.shape[1]
model = Sequential([
    LSTM(256, activation = 'relu', input_shape = (None, 1)),
    Dropout(0.1),
    Dense(128, activation = 'relu'),
    Dropout(0.1),
    Dense(outputs)
])

model.compile(loss = 'mse', optimizer = Adam(lr = 1e-3), metrics = ["mae"])
earlystop = EarlyStopping(patience = 32, monitor = 'val_loss', mode = 'auto', restore_best_weights = True)
cb = [PlotLossesCallback(), earlystop]
model.fit(
    des_xtrain,
    des_ytrain,
    validation_split = 0.2,
    epochs = 100,
    shuffle = True,
    batch_size = 16,
    verbose = 1,
    callbacks = cb
)

ypred = model.predict(des_xtest)
for i in np.arange(ypred.shape[0] - 1, -1, -1):
    ypred[ypred.shape[0]-1-i] += res.seasonal[-(13 + i):-(1 + i)]
    
plot_eval(ypred, ytest, n = 12)