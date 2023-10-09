import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from keras import regularizers

warnings.filterwarnings('ignore')

data_train=pd.read_csv("poker-hand-training-true.data",header=None)
data_test = pd.read_csv("poker-hand-testing.data",header=None)
col=['Suit of card #1','Rank of card #1','Suit of card #2','Rank of card #2','Suit of card #3','Rank of card #3','Suit of card #4','Rank of card #4','Suit of card #5','Rank of card 5','Poker Hand']
data_train.columns=col
data_test.columns=col
y_train=data_train['Poker Hand']
y_test=data_test['Poker Hand']
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)
x_train=data_train.drop('Poker Hand',axis=1)
x_test=data_test.drop('Poker Hand',axis=1)
print('Shape of Training Set:',x_train.shape)
print('Shape of Testing Set:',x_test.shape)

model = Sequential()
model.add(Dense(15, activation='relu', input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs = 10, batch_size = 256, verbose=1,validation_data=(x_test,y_test),shuffle=True)


score = model.evaluate(x_test, y_test, batch_size=256)
