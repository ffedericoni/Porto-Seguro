"""
Created on Tue Oct 24 14:36:52 2017

@author: NF299

Join submission files into a Dataframe and build a model to anticipate 
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
count= 0 #just to go quicket while prototyping
for item in os.listdir():
    count+=1                #TODO: to be removed
    if count > 4: break     #TODO: to be removed
    if item.endswith(extension):
        filename = os.path.abspath(item)
        
        dashpos = filename.index('-')
        score = filename[dashpos+1:dashpos+4]
        print(filename, score)
        target = pd.read_csv(filename)

        preds = target['target']  # preds is a Series
        preds = preds.sort_values() 
        preds = preds.reset_index()
        preds = preds.drop(['index'], axis=1) # reindexed according to sort order
        X_tr = preds.T
        X_tr['score'] = score
        X_tr['filename'] = item
        X = X.append(X_tr)

X = X.set_index('filename')
print("Finished Reading files", X.shape)

# =============================================================================
# X['cum'] = pd.Series(data='NaN', index=X.index, dtype='float64')
# X.iloc[0, 1] = 0 #initialze the cumulation
# for ind in range(len(X)):
#     if ind == 0:
#         continue
#     X.iloc[ind, 1] = X.iloc[ind - 1, 1] + X.iloc[ind - 1, 0]
# =============================================================================


# =============================================================================
# At this point X is like this:
#    target      cum
# 0  0.00346  0.00000
# 1  0.00374  0.00346
# 2  0.00376  0.00720
# 3  0.00402  0.01096
# 4  0.00408  0.01498
# =============================================================================


# =============================================================================
# regr = LinearRegression()
# y = X['target']
# X = X.drop('target', axis=1)
# print('(', X.shape, y.shape, ')')
# regr = regr.fit(X, y)
# 
# item=56
# 
# pred = regr.predict(X[item])
# 
# print (pred, y[item])
# 
# =============================================================================



        