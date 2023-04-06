import datetime
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Sequential
import requests
from datetime import datetime
from datetime import timedelta
import pandas as pd

BASE_URL = "https://api.gateio.ws"
CONTEX = "/api/v4"
URL = '/spot/candlesticks'
SYMBOL = 'BTC_USDT'
INTERVAL = '1m'
LIMIT = '1000'



class class_f:
    @classmethod
    def rest_api_get_candlesticks(cls, base_url, contex, url, symbol, interval, limit):
        loading = requests.get(base_url+contex+url, 
                                params = {'currency_pair' : f'{symbol}', 'interval' : f'{interval}', 'limit' : f'{limit}'})
        loading = loading.json()
        return loading
        
    @classmethod
    def close(cls, r):
        #
        close_long =list()
        index_time_long = list()
        #
        for kline in r:
            close_long.append(float(kline[2])) 
            time_long = datetime.fromtimestamp(int(kline[0])).strftime("%Y-%m-%d %H:%M:%S")   
            index_time_long.append(time_long)   #  time  in  int()
            #
        return [close_long,  index_time_long]

    @classmethod
    def data_frame(cls, series, time, columns):
        i=0
        ts = list()
        for t in time:
            ts.append(t)
            ts.append(series[i])
            i += 1
        close = pd.Series(series)
        close.index = pd.to_datetime(time)   
        data_f = pd.DataFrame (close, columns = [columns])
        return data_f

    @classmethod
    def model_bidirectional_lstm_elu(cls, units: int(), x_train, prediction_minutes):
      model = Sequential ([ 
            Bidirectional( LSTM(units = units, activation = "elu", return_sequences=True, go_backwards=True,  stateful=False, 
                                            input_shape=(x_train.shape[1], 1)) ), 
            Dropout(.05),
            Bidirectional( LSTM(units = units*4, activation = "softsign", return_sequences=True, go_backwards=True) ),
            Dropout(.1),
            Bidirectional( LSTM(units = units*8, activation = "elu", return_sequences=True, go_backwards=True) ),
            Dropout(.1),
            Bidirectional( LSTM(units = units*2, activation = "LeakyReLU", return_sequences=True, go_backwards=True) ),   #  tanh
            Dropout(.1),
            #
            Bidirectional( LSTM(units = units*4, activation = "elu", return_sequences=True, go_backwards=True) ),
            Dropout(.1),
            #
            Bidirectional(LSTM(units = units)),
            Dropout(.05),
            Dense(prediction_minutes)
      ]) 
      return model 

    @classmethod
    def model_bidirectional_lstm_tanh(cls, units: int(), x_train, prediction_minutes):
      model = Sequential ([ 
            Bidirectional( LSTM(units = units, activation = "tanh", return_sequences=True, go_backwards=True,  stateful=False, 
                                            input_shape=(x_train.shape[1], 1)) ), 
            Dropout(.1),
            Bidirectional( LSTM(units = units*16, activation = "tanh", return_sequences=True, go_backwards=True) ),
            Dropout(.5),
            Bidirectional( LSTM(units = units, activation = "tanh", return_sequences=True, go_backwards=True) ),
            Bidirectional( LSTM(units = units, activation = "tanh", return_sequences=True, go_backwards=True) ),
            Bidirectional( LSTM(units = units*32, activation = "relu", return_sequences=True, go_backwards=True) ),
            Dropout(.5),
            Bidirectional( LSTM(units = units, activation = "tanh", return_sequences=True, go_backwards=True) ),
            Bidirectional( LSTM(units = units*4, activation = "linear", return_sequences=True, go_backwards=True) ),
            Dropout(.5),
            Bidirectional( LSTM(units = units, activation = "tanh", return_sequences=True, go_backwards=True) ),
            #
            Bidirectional( LSTM(units = units*4, activation = "linear", return_sequences=True, go_backwards=True) ),
            Bidirectional( LSTM(units = units*16, activation = "tanh", return_sequences=True, go_backwards=True) ),
            Dropout(.5),
            #
            Bidirectional(LSTM(units = units)),
            Dropout(.1),
            Dense(prediction_minutes)
      ]) 
      return model

###   END
