import finnhub
from datetime import datetime, timedelta
from pandas_datareader import data
import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

pasttDate=datetime(2013, 5, 1, 10, 0)
dt=datetime.timestamp(pasttDate)*1000
print(dt)
ts=int('1367402400')
print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

presentDate=datetime.now()
dt=datetime.timestamp(presentDate)*1000
print(dt)
dc=int(dt)
type(dc)

finnhub_client = finnhub.Client(api_key="cip5aphr01qrdahju3f0cip5aphr01qrdahju3fg")
res = finnhub_client.stock_candles('MSFT', 'D', 1367402400 ,dc )
print(res)

import pandas as pd
df=pd.DataFrame(res)
print(df)

df['t'] = pd.to_datetime(df['t'], unit='s')
print(df)

df1=df.reset_index()['o']
print(len(df1))

scaler=MinMaxScaler(feature_range=(0,1))#data preprocessing LSTM are sensitive to scale of the data  #values b/w 0and 1
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
print(df1)

from keras.models import load_model

model=load_model("/content/Finnhublstm_model.h5")

scaler=MinMaxScaler(feature_range=(0,1))#data preprocessing LSTM are sensitive to scale of the data  #values b/w 0and 1
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
print(df1)

test_data=df1[0:len(df1),:1]
print(len(test_data))

x=len(test_data)-100
print(x)

x_input=test_data[x:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0
predictions = {}
prediction_data = []
dtc=(df['t'].tail(1))
while i<2:
  if(len(temp_input)>100):
    #print(temp_input)
    x_input=np.array(temp_input[1:])
    # print("{} day input {}".format(i,x_input))
    x_input=x_input.reshape(1,-1)
    x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
    yhat = model.predict(x_input, verbose=0)
    y_hat = yhat
    y_hat = scaler.inverse_transform(y_hat)
    #print("{} day output {}".format(pd.to_datetime(dtc)+timedelta(i+2),y_hat))
    #predictions ={"date": pd.to_datetime(dtc) + timedelta(days=i+1), "prediction": y_hat}
    predictions_list = y_hat.tolist()
    date_strings = pd.to_datetime((dtc)+ timedelta(days=i)).dt.strftime('%Y-%m-%d').tolist()

    for date_str, prediction in zip(date_strings, predictions_list):
      prediction_dict = {"date": date_str, "prediction": prediction}
      prediction_data.append(prediction_dict)
      print(prediction_data)


    temp_input.extend(yhat[0].tolist())
    temp_input=temp_input[1:]
    #print(temp_input)
    lst_output.extend(yhat.tolist())
    i=i+1
  else:
    x_input = x_input.reshape((1, n_steps,1))
    yhat = model.predict(x_input, verbose=0)
    # print(yhat[0])
    temp_input.extend(yhat[0].tolist())
    # print(len(temp_input))
    lst_output.extend(yhat.tolist())
    i=i+1