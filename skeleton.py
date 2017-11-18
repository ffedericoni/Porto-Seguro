# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:47:00 2017

@author: NF299
"""

"""
Skeleton for a program to participate to Kaggle competition
"""

import ff_util as ff

Competition = dict()
Competition['name'] = 'Porto Seguro'

start = ff.timer()
train, test = ff.read_kaggle_data(Competition['name'])
ff.timer(start)

print("Train Shape=", train.shape)
print("Test Shape=", test.shape)

print("Train types=", train.dtypes)

train = ff.reduce_memory_footprint(train)
test = ff.reduce_memory_footprint(test)

print("Train types=", train.dtypes)

#feature selection

#features engineering

#features scaling, if necessary

#parameter tuning wth cross validation

#training

#evaluate on test set

#submit results

