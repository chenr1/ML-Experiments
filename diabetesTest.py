# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:43:58 2022

@author: LabUser
"""

# example of evaluating a knn model on the diabetes classification dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# load the dataset
url = 'training.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define model
model = KNeighborsClassifier()
# fit model
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy * 100))