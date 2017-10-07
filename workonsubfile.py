# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:31:20 2017

@author: NF299
"""

#Manipulate the submission file
import pandas as pd

sub = pd.read_csv('subfile.csv')
print(sub.describe())

conditionUP = sub > 0.06
print(sub[conditionUP].describe())

sub['target'][conditionUP] = sub['target'][conditionUP] * 1.01
print('--------------')
print(sub.describe())
print(sub[conditionUP].describe())

sub.to_csv('Pump280.csv', index=False)

