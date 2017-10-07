# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 19:08:47 2017

@author: NF299
"""
# In the competition context, (Kaggle Porto Seguro)
# we are evaluating the concentration of claims against the distribution of score
# (For example, the top 5% drivers by score have 50% of all insurance claims).
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/40222
# Column names:
# "Ind" is related to individual or driver, "reg" is related to region,
# "car" is related to car itself and "calc" is an calculated feature.
# "reg" features represent qualities of a regions on continuous/ordinal scale.
# ps_car_15 are square root of integers, 0-14

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import gini
import time

start_time = time.time()

df_train = pd.read_csv('../input/train.csv')
#train_stat = df_train.iloc[:,2:].describe().T
df_test  = pd.read_csv('../input/test.csv')
#print(df_train.head())
print("Data read in ", time.time() - start_time )

minix = 0
maxix = df_train.shape[0] #500000

#TRAIN SETs
#target column
y_train = df_train.iloc[minix:maxix,1]
#57 features
X_train = df_train.iloc[minix:maxix,2:]

#TEST SETs
#57 features
X_test = df_test.drop(['id'],axis = 1)

#print(X.shape(), y.shape())

#clf = LinearSVC()
clf = RandomForestClassifier(n_estimators = 100,
                             min_samples_leaf = 10,
                             min_samples_split=15,
                             class_weight='balanced',
                             random_state = 42)
clf.fit(X_train, y_train) 
y_test = clf.predict(X_test[minix:maxix])
print("RF Score=", clf.score(X_train, y_train) )
print("Features Importance=", df_train.columns[2:][clf.feature_importances_<0.005])
print("Model built in ", time.time() - start_time)
print("Score Gini=", gini.gini(y_train, y_test) )
print('Gini calculated in ', time.time()  - start_time)

#y_test = y_test[:,1]
#sample_submission = pd.read_csv("../data/sample_submission.csv")
#sample_submission["target"] = 0
#sample_submission.set_index("id", inplace=True)
#sample_submission.to_csv("submission_empty.csv")



