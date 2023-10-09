# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:28:20 2022

@author: LabUser
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from xgboost import XGBClassifier
import joblib

test = pd.read_csv('poker-hands-testing-set.csv', header=None)
train = pd.read_csv('poker-hands-testing-set.csv', header=None)
train.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']
test.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']

# train.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']
# test.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']
train.head()
train.shape
print(train.shape)

X_train = train.loc[:,train.columns != 'Label']
X_test = test.loc[:,test.columns != 'Label']
Y_train = train['Label']
Y_test = test['Label']


Y_train.groupby(Y_train).size()
Y_test.groupby(Y_test).size()

def preprocess_data(data:pd.DataFrame):
    df = data.copy()
    dfc = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    dfc.values.sort()
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = dfc
    df = df[['C1', 'C2', 'C3', 'C4', 'C5', 'S1', 'S2', 'S3', 'S4', 'S5', 'Label']]
    return df

def resortData(data:pd.DataFrame):
    df = data.copy()
    dfc = df[['Card1', 'Card2', 'Card3', 'Card4', 'Card5']]
    dfc.values.sort()
    df[['Card1', 'Card2', 'Card3', 'Card4', 'Card5']] = dfc
    df = df[['Card1', 'Card2', 'Card3', 'Card4', 'Card5', 'Suit1', 'Suit2', 'Suit3', 'Suit4', 'Suit5', 'Category']]
    return df

    



X_train_pre = preprocess_data(train)
X_test_pre = preprocess_data(test)

print(f"xTrain notprocessed: \n {X_train.head()}")
print(f"xTrain processed: \n {X_train_pre.head()}")


X_train = X_train_pre.loc[:,X_train_pre.columns != 'Label']
X_test = X_test_pre.loc[:,X_test_pre.columns != 'Label']

alg = DecisionTreeClassifier(random_state=1)
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)




filename = 'finalized_model.sav'
joblib.dump(alg, filename)

print(accuracy_score(Y_test, y_pred, normalize=True))



