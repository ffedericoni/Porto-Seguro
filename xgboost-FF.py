# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 18:36:22 2017

@author: NF299
"""
#copy of Kernel https://www.kaggle.com/kueipo/stratifiedshufflesplit-xgboost-example-0-28/code

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#import lightgbm as lgb
#import xgboost as xgb
import time

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


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
#li_features = [] #Dont drop
train = train.drop(li_features, axis=1)
test = test.drop(li_features, axis=1)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

#Multiply the most important features (indexes are based on not dropped features)
mi_features = [35, 14, 2, 20, 34]
mi_features = []  #No feature engineering
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


# A parameter grid for XGBoost
params = {
        'min_child_weight': [3, 4, 5],
        'gamma': [1, 1.25, 1.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [5]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=800, objective='binary:logistic',
                    silent=True, missing=-1)
folds = 5
param_comb = 12

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 5)
sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.30, random_state=5)

random_search = RandomizedSearchCV(xgb, param_distributions=params, 
                                   n_iter=param_comb, scoring='roc_auc', 
                                   n_jobs=4, cv=sss.split(X,y), 
                                   verbose=4, random_state=101, 
                                   iid=True )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y, early_stopping_rounds=70)
timer(start_time) # timing ends here for "start_time" variable

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
y_test = random_search.predict_proba(test.values)
results_df = pd.DataFrame(data={'id':id_test, 'target':y_test[:,1]})
filename = 'RGS-xgb-d' + str(start_time.day) + '-h' + str(start_time.hour) + '.csv'
results_df.to_csv(filename, index=False)


