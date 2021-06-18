# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:50:07 2021

@author: cco24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

f = open('samsung_stock.csv', 'r')
stock_data = pd.read_csv(f, header = 0)
seq = stock_data[['Close']].to_numpy()

def seq2dataset(seq, window, horizon):
    X=[];Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=seq[i+window+horizon-1]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

w=3
h=1

X,Y = seq2dataset(seq, w, h)

split = int(len(X)*0.7)
x_train = X[0:split]
y_train = Y[0:split]
x_test = X[split:]
y_test = Y[split:]

lstm = Sequential()
lstm.add(LSTM(units=512, activation='relu', input_shape=x_train[0].shape))
lstm.add(Dense(1))
lstm.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist = lstm.fit(x_train, y_train, epochs=200, batch_size=1
                 ,validation_data=(x_test,y_test), verbose = 2)

ev = lstm.evaluate(x_test, y_test, verbose=0)
print("손실 함수:", ev[0],"MAE:",ev[1])

pred = lstm.predict(x_test)
print("평균절댓값백분율오차(MAPE):", sum(abs(y_test-pred)/y_test)/len(x_test))


plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()

x_range=range(len(y_test))
plt.plot(x_range, y_test[x_range], color = 'red')
plt.plot(x_range, pred[x_range], color = 'blue')
plt.legend(['True Prices', 'Predicted Prices'], loc='best')
plt.grid()
plt.show()

lstm.save("samsung_stock_lstm.h5")

