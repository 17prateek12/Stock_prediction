from flask import Flask, jsonify, request
import json
import finnhub
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

api = Flask(__name__)
api.debug = True


@api.route('/predict' ,methods=['POST'])
def prediction():
    data=request.json
    stockSymbol=data.get('stockSymbol')
    if not stockSymbol:
        return jsonify({'error': 'No stock symbol provided.'}), 400
    presentDate=datetime.now()
    dt=datetime.timestamp(presentDate)*1000
    dc=int(dt)
    finnhub_client = finnhub.Client(api_key="cip5aphr01qrdahju3f0cip5aphr01qrdahju3fg")
    res = finnhub_client.stock_candles(stockSymbol, 'D', 1367402400 ,dc )
    import pandas as pd
    df=pd.DataFrame(res)
    df1=df.reset_index()['o']
    df['t'] = pd.to_datetime(df['t'], unit='s')
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    test_data=df1[0:len(df1),:1]
    x=len(test_data)-100
    x_input=test_data[x:].reshape(1,-1)
    model=load_model("D:/web development/stock web app/backend/Finnhublstm_model.h5")
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
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        y_hat = yhat
        y_hat = scaler.inverse_transform(y_hat)
        predictions_list = y_hat.tolist()
        date_strings = pd.to_datetime((dtc)+ timedelta(days=i)).dt.strftime('%Y-%m-%d').tolist()
        
        for date_str, prediction in zip(date_strings, predictions_list):
          prediction_dict = {"date": date_str, "prediction": prediction}
          prediction_data.append(prediction_dict)
      
        
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
    #data=predictions.copy

    return jsonify(prediction_data)

    