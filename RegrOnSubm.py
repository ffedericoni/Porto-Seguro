"""
Created on Tue Oct 24 14:36:52 2017

@author: NF299

Join submission files into a Dataframe and bild a model to anticipate 
the LeaderBoard score of new submissions
"""

print(__doc__)

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

extension = ".csv"
dir_name = "C:\\Users\\NF299\\Documents\\Python\\Kaggle\\Porto Seguro\\submissions\\CSV"
os.chdir(dir_name)

X = pd.DataFrame()

for item in os.listdir():
    if item.endswith(extension):
        filename = os.path.abspath(item)
        
        dashpos = filename.index('-')
        score = filename[dashpos+1:dashpos+4]
        print(filename, score)
        target = pd.read_csv(filename)
#        X_tr = target.T
        X = target['target']  # X is a Series
        X = X.sort_values() 
        X = X.reset_index()
        X = X.drop(['index'], axis=1) # reindicizzato secondo il sort
#        X_tr['target'] = score
#        X = X.append(X_tr)


X['delta'] = pd.Series(data='NaN', index=X.index, dtype='float64')
for ind in range(len(X)-1):
    X.iloc[ind, 1] = X.iloc[ind+1, 0] - X.iloc[ind, 0]


regr = LinearRegression()
y = X['target']
X = X.drop('target', axis=1)
print('(', X.shape, y.shape, ')')
regr = regr.fit(X, y)

item=56

pred = regr.predict(X[item])

print (pred, y[item])




        