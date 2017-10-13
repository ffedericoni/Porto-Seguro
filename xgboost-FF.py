# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 18:36:22 2017

@author: NF299
"""
#copy of Kernel https://www.kaggle.com/kueipo/stratifiedshufflesplit-xgboost-example-0-28/code

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#import lightgbm as lgb
import xgboost as xgb
import time

# Read in our input data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# This prints out (rows, columns) in each dataframe
print('Train shape:', train.shape)
print('Test shape:', test.shape)

#Add positive to help balancing
#positive = train[train['target'] == 1]
#train = train.append(positive).append(positive).append(positive)
#train = train.append(positive).append(positive).append(positive)


y = train.target.values
id_test = test['id'].values


# We drop these variables as we don't want to train on them

start_time=time.time()
train = train.drop(['id','target'], axis=1)
test = test.drop(['id'], axis=1)

#ff drop least important features
#worst 5
li_features = ['ps_ind_13_bin', 'ps_ind_10_bin',
'ps_ind_11_bin', 'ps_ind_14', 'ps_ind_12_bin']
#second worst 5
li_features += ['ps_car_10_cat', 'ps_calc_20_bin',
'ps_ind_18_bin', 'ps_calc_15_bin', 'ps_calc_16_bin']
li_features = [] #Dont drop
train = train.drop(li_features, axis=1)
test = test.drop(li_features, axis=1)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

#Multiply the most important features (indexes are based on not dropped features)
mi_features = [35, 14, 2, 20, 34]
mi_feature_names = ['ps_car_14', 'ps_ind_15', 'ps_ind_03',
'ps_reg_03', 'ps_car_13']
name = []
for ind in range(len(mi_features)):
    im = mi_features[ind]
    for indi in range(len(mi_features)):
        ik = mi_features[indi]
        if ind < indi:
            name = 'm_' + str(im) + str(ik)
            print(name)
            train[name] = train.iloc[:,im] * train.iloc[:,ik]
            test[name] = test.iloc[:,im] * test.iloc[:,ik]
            
print('Train shape:', train.shape)
print('Test shape:', test.shape)

X = train.values

# Set xgb parameters

params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.03
params['silent'] = True
params['max_depth'] = 5  #ff era 5
params['subsample'] = 0.9
params['colsample_bytree'] = 0.85
params['colsample_bylevel'] = 0.9
params['eval_metric'] = 'auc'
#params['tree_method'] = 'exact'

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = np.zeros_like(id_test)

# Take a random 30% of the dataset as validation data

kfold = 5
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.15, random_state=15)
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
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, #feval=gini_xgb,
                    maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediction:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test)
    sub['target'] += p_test/kfold


# Create a submission file
sub.to_csv('StratifiedShuffleSplit.csv', index=False)
print('Elapsed Time =', time.time() - start_time)
print('Best Score=', mdl.attr('best_score') )

#Features importance
import operator
x = mdl.get_fscore()
sorted_score = sorted(x.items(), key=operator.itemgetter(1))
for ind in range(len(sorted_score)):
    ii = int(sorted_score[ind][0].replace('f', ''))
    print(sorted_score[ind]+(train.columns[ii],))
#print('Features Score=', sorted_score)
#for i,f in enumerate(train.columns):
#    print(i,f)
