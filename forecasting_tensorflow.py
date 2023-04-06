#!/usr/bin/env python3.8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import copy 
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from functions import class_f as _
from functions import BASE_URL, CONTEX, URL, SYMBOL, INTERVAL, LIMIT

######################################################################
#
#   LOADING DATA
#
######################################################################

long = _.rest_api_get_candlesticks(BASE_URL, CONTEX, URL, SYMBOL, INTERVAL, LIMIT)
close_l, index_time_l = _.close(long)

data_long_first= _.data_frame(close_l, index_time_l, "Long")

data_long = data_long_first[:-60]
# data_long['Date'] = data_long.index  
data_long.loc[:,['Date']] = data_long.iloc[:].index    
data_long.index = data_long['Date']
data_long = data_long.drop('Date', axis=1)
data_long.plot()
plt.show()


######################################################################
#
#   DATA TRANSFORMATION FOR TRAINING
#
######################################################################

data = data_long
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_long['Long'].values.reshape(-1,1))
prediction_minutes  =  60

x_train, y_train = [], []

for x in range(prediction_minutes, len(scaled_data)):
    x_train.append(scaled_data[ x-prediction_minutes : x, 0 ])
    y_train.append(scaled_data[ x, 0 ])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1], 1) )


######################################################################
#
#   MODEL BUILDING
#
######################################################################

pd.options.mode.chained_assignment = None
np.random.seed(0)  #  " Seed must be between 0 and 2**32 - 1  =  4294967295"
tf.random.set_seed(0)   # 959774032

model = _.model_bidirectional_lstm_tanh(units = 2, x_train=x_train, prediction_minutes=prediction_minutes)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=300000,
    decay_rate=0.97,
    staircase=False)
 
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="mse", metrics=['mean_squared_error'])


######################################################################
#
#   TRAIN
#
######################################################################

filepath = "/home/pokoj/Pulpit/PYTHON/GATE_IO/weights.best_epochs_.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'mean_squared_error', 
                                                verbose=1, save_best_only = True, mode = 'min') 
es = EarlyStopping(monitor='mean_squared_error', patience=1)        
callbacks_list = [checkpoint, es]

repetition_of_experiments = 11
for i in range(repetition_of_experiments):
    model.fit(x_train, y_train, epochs=5, batch_size=512, callbacks=callbacks_list, 
                                                    shuffle=True, verbose=1,validation_split=.2) 
    model.fit(x_train, y_train, epochs=1, batch_size=16, shuffle=True, verbose=1, 
                                                    validation_split=.05)                                                    


######################################################################
#
#   TEST
#
######################################################################

test_data = data.iloc[500:]
actual_prices = test_data['Long'].values
total_dataset = pd.concat((data['Long'], test_data['Long']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_minutes : ].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_minutes, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_minutes : x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test) 
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color = 'black', label = 'Actual  Prices')
plt.plot(prediction_prices, color = 'green', label = 'Predicted  Prices')
plt.title("  BTC  price   prediction  ")
plt.xlabel(' Time ')
plt.ylabel(' Price ')
plt.legend(loc='upper left')
plt.show()


######################################################################
#
#  FORECASTING
#
######################################################################

real_data = [model_inputs[len(model_inputs) + 1 - prediction_minutes : len(model_inputs) + 1,  0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

forecast = model.predict(real_data)
forecast = scaler.inverse_transform(forecast)
forecast = forecast[0]


######################################################################
#
#  RESULT TRANSFORMATION TO A DATA FRAME
#
######################################################################

df_past = data_long[['Long']].reset_index()
df_past.rename(columns={'index': 'Date', 'Long': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].values[-1] = df_past['Actual'].values[-1].copy()

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(minutes=1), periods=prediction_minutes, freq ='1T')
df_future['Forecast'] = forecast.flatten()
df_future['Actual'] = np.nan

df = pd.concat([df_past, df_future], axis=0, ignore_index=True)
df.index = df['Date']
del(df['Date'])
results = df

results.plot(title='BTC')
plt.legend()
plt.show()
#############################
df_future.index = df_future['Date']
plt.plot(df_future['Forecast'], label = 'forecast')
plt.plot(data_long_first, label = 'actual')
plt.legend()
plt.show()


######################################################################
#
#  BOLLINGER BANDS CHART FOR FORECAST ERROR
#
######################################################################
p_m = prediction_minutes
results[ 'Actual'].values[-p_m: ] = data_long_first['Long'].values[-p_m: ].copy()
results['Forecast_error'] = (results['Forecast'] - results['Actual']).values 
print(' \n'* 2 )
print(results[-p_m:])
print(' \n'* 2 )
print(results[['Forecast_error']].describe(include='all'))
#############################
period = 13
multiplier = 1.5
df_future['Upper_std'] = df_future['Forecast'].rolling(period).mean() + df_future['Forecast'].rolling(period).std() * multiplier
df_future['Lower_std'] = df_future['Forecast'].rolling(period).mean() - df_future['Forecast'].rolling(period).std() * multiplier

plt.plot(df_future['Forecast'], label = 'forecast', color = 'blue', marker = 'o', markerfacecolor = 'b')
plt.plot(df_future['Upper_std'], label = 'upper band', color = 'red')
plt.plot(df_future['Lower_std'] , label = 'lower band', color = 'green')
plt.plot(data_long_first[-p_m: ], label = 'actual', color = 'black', marker = 'o', markerfacecolor = 'k')
plt.legend()
plt.show()
#############################
#   END
