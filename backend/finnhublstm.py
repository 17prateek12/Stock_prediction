


import finnhub
from datetime import datetime

ts=int('1690177692')
print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

ts=int('1367402400')
print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

pasttDate=datetime(2013, 5, 1, 10, 0)
dt=datetime.timestamp(pasttDate)*1000
print(dt)

presentDate=datetime.now()
dt=datetime.timestamp(presentDate)*1000
print(dt)
dc=int(dt)
type(dc)

finnhub_client = finnhub.Client(api_key="cip5aphr01qrdahju3f0cip5aphr01qrdahju3fg")
res = finnhub_client.stock_candles('GOOG', 'D', 1367402400 ,dc )
print(res)

import pandas as pd
df=pd.DataFrame(res)
print(df)

x=len(df['t'])
print(x)

df['t'] = pd.to_datetime(df['t'], unit='s')
print(df)

df1=df.reset_index()['o']

from pandas_datareader import data
import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from datetime import datetime as dt

plt.plot(df1)

ma100 = df.o.rolling(100).mean()
ma200 = df.o.rolling(200).mean()
ma100,ma200

plt.figure(figsize = (24 ,12))
plt.plot(df.o)
plt.plot(ma100,'g', label ='ma 100')
plt.plot(ma200,'r',label ='ma 200')

scaler=MinMaxScaler(feature_range=(0,1))#data preprocessing LSTM are sensitive to scale of the data  #values b/w 0and 1
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
print(df1)

training_size=int(len(df1)*0.75)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
training_size,test_size
type(test_data)

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1) #converting it into 3d
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))#activation function -  relu
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=150,batch_size=64,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

model.save("Finnhublstm_model.h5")

print(train_predict)

print(test_predict)

y_testreal = ytest
y_testreal = y_testreal.reshape(-1, 1)
y_testreal=scaler.inverse_transform(y_testreal)
print(y_testreal)

math.sqrt(mean_squared_error(y_train,train_predict))

math.sqrt(mean_squared_error(ytest,test_predict))

look_back=100

trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.figure(figsize = (24 ,6))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(train_data)

len(test_data)

x=len(test_data)-100
print(x)

x_input=test_data[x:].reshape(1,-1)
x_input.shape

x_input.shape[1]

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0

while(i<10):
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
    print("{} day output {}".format((i+2),y_hat))
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

import pandas as pd
from datetime import timedelta
i=1
dt=(df['t'].tail(1))
print(dt)
dc=pd.to_datetime(dt)+timedelta(i)
print(dc)

day_new=np.arange(1,101)
day_pred=np.arange(101,111)

plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]),label='Trained data')
plt.plot(day_pred, scaler.inverse_transform(lst_output),label='Next 30 days')

#acc
x = math.sqrt(mean_squared_error(y_train,train_predict))
x

type(x)

a = 522.6069 + x
acc= (x/a)*100
acc = 100- acc
print('The accuracy is :')
print(acc)

"""# New Section"""