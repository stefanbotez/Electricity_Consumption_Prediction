import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dropout, Conv1D, Activation, MaxPooling1D, Flatten
from pandas import Grouper
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, ReLU, LSTMCell, \
    BatchNormalization, Dropout, LSTM, Conv1D, Activation, MaxPool1D, Flatten

# TRAIN_BARRIER = 392150 # ultimele 6 luni pt validare
TRAIN_BARRIER = 365905 # ultimele 12 luni pt validare
traind = pd.read_csv("train_electricity.csv")

def remove_aberrant(dat):
    dat = dat.loc[np.invert(dat['Coal_MW'] > 4500) | np.invert(dat['Coal_MW'] < 300), :]
    dat = dat.loc[np.invert(dat['Gas_MW'] < 30), :]
    dat = dat.loc[np.invert(dat['Nuclear_MW'] < 400), :]
    dat = dat.loc[np.invert(dat['Biomass_MW'] > 80), :]
    dat = dat.loc[np.invert(dat['Production_MW'] < 3000), :]
    dat = dat.loc[np.invert(dat['Consumption_MW'] > 11000) | np.invert(dat['Consumption_MW'] < 2000), :]
    return dat


traind = remove_aberrant(traind)

traind = traind.sort_values(by=["Date"], ascending=True)

train_data = traind.drop(columns=['Consumption_MW', 'Date'])
train_labels = traind['Consumption_MW']


sc = MinMaxScaler(feature_range=(0, 1))

train_data = sc.fit_transform(train_data)
train_labels = train_labels.values[:, None].astype(np.float)

nr_in = train_data.shape[1]
BACK = 100


def process_data(dat, labels):
    x = []
    y = []
    for i in range(BACK, dat.shape[0], 4):
        x.append(np.reshape(dat[i - BACK:i], (BACK, nr_in,)))
        y.append(labels[i])
    return np.array(x), np.array(y)


train_data, train_labels = process_data(train_data, train_labels)

model = Sequential()
model.add(Conv1D(filters=500, kernel_size=2, input_shape=(BACK, nr_in,)))
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Conv1D(filters=250, kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Conv1D(filters=50, kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

model.summary()
history = model.fit(train_data, train_labels, epochs=50, batch_size=64, validation_split=0.1264)

print(np.min(history.history['val_loss']))

model.save_weights("model.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
