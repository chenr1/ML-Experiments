# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:28:20 2022

@author: LabUser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import joblib
import seaborn as sns
from collections import Counter


#load in files for training and testing
dfTest = pd.read_csv('poker-hands-test-set.csv', header=None)
dfTrain = pd.read_csv('poker-hands-training-set.csv', header=None)

#the data comes with no columns, lets add them in
dfTrain.columns = ['Suit1', 'Card1','Suit2', 'Card2','Suit3', 'Card3','Suit4', 'Card4','Suit5', 'Card5','Category']
dfTest.columns = ['Suit1', 'Card1','Suit2', 'Card2','Suit3', 'Card3','Suit4', 'Card4','Suit5', 'Card5','Category']

#remove the category column
X_train = dfTrain.loc[:,dfTrain.columns != 'Category']
X_test = dfTest.loc[:,dfTest.columns != 'Category']
Y_train = dfTrain['Category']
Y_test = dfTest['Category']

def resortData(data):
    sortedDF = data.copy()
    dfc = sortedDF[['Card1', 'Card2', 'Card3', 'Card4', 'Card5']]
    dfc.values.sort()
    sortedDF[['Card1', 'Card2', 'Card3', 'Card4', 'Card5']] = dfc
    sortedDF = sortedDF[['Card1', 'Card2', 'Card3', 'Card4', 'Card5', 'Suit1', 'Suit2', 'Suit3', 'Suit4', 'Suit5', 'Category']]
    return sortedDF

    

X_train_pre = resortData(dfTrain)
X_test_pre = resortData(dfTest)


X_train = X_train_pre.loc[:,X_train_pre.columns != 'Category']
X_test = X_test_pre.loc[:,X_test_pre.columns != 'Category']

#train and test the model on the Training and Test set appropriately
alg = DecisionTreeClassifier(random_state=1)
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)

#save the trained model
filename = 'finalized_model.sav'
joblib.dump(alg, filename)
print(f"File saved under {filename}")
#output final model accuracy score
finalScore = accuracy_score(Y_test, y_pred, normalize=True)
print(f"Model has accuracy of {finalScore}")

#VISUAL THINGS
#compare results
pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['True'] = true_series
pred_res['Predicted'] = pred_series
print(pred_res)
f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Category', y='Count', hue='Variable')

