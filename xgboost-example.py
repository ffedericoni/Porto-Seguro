# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 18:36:22 2017

@author: NF299
"""
#copy of Kernel https://www.kaggle.com/kueipo/stratifiedshufflesplit-xgboost-example-0-28/code

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#import lightgbm as lgb
import xgboost as xgb
import time

# Read in our input data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# This prints out (rows, columns) in each dataframe
print('Train shape:', train.shape)
print('Test shape:', test.shape)


y = train.target.values
id_test = test['id'].values


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)

# Create an XGBoost-compatible metric from Gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score
    
# We drop these variables as we don't want to train on them
# The other 57 columns are all numerical and can be trained on without preprocessing

start_time=time.time()
train = train.drop(['id','target'], axis=1)
test = test.drop(['id'], axis=1)
#ff drop least important features
li_features = ['ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',
       'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_18_bin',
       'ps_car_02_cat', 'ps_car_08_cat', 'ps_car_10_cat', 'ps_calc_15_bin',
       'ps_calc_20_bin']
train = train.drop(li_features, axis=1)
test = test.drop(li_features, axis=1)

print('Train shape:', train.shape)
print('Test shape:', test.shape)


X = train.values

# Set xgb parameters

params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.03
params['silent'] = True
params['max_depth'] = 6  #ff era 5
params['subsample'] = 0.9
params['colsample_bytree'] = 0.85
params['colsample_bylevel'] = 0.9
#params['tree_method'] = 'exact'

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = np.zeros_like(id_test)

# Take a random 30% of the dataset as validation data

kfold = 5
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.15, random_state=42)
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # Convert our data into LGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 2,000 rounds (with early stopping after 100)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=90, feval=gini_xgb, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediction:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test)
    sub['target'] += p_test/kfold



# Create a submission file
sub.to_csv('StratifiedShuffleSplit.csv', index=False)
print('Elapsed Time =', time.time() - start_time)
print('Best Score=', mdl.attr('best_score') )
#Features importance
#mdl.get_fscore()
#for i,f in enumerate(train.columns):
#    print(i,f)
