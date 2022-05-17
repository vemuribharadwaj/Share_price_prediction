# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:13:19 2018

@author: Abhishek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
train_set=dataset_train.iloc[:,1:2]

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
fit=sc.fit(train_set)
scaled_training=sc.fit_transform(train_set)


x_train=[]
y_train=[]

for i in range(60,1258):
    x_train.append(scaled_training[i-60:i,0])
    y_train.append(scaled_training[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


model= Sequential()

model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
0

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs = 100, batch_size = 32)


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


dataset_total = pd.concat(( dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

    
