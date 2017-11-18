# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:24:11 2017

@author: ffedericoni
"""

print(__doc__)
import pandas as pd
from datetime import datetime

# Set skeleton_test = False if you need to chdir ..
skeleton_test = True


def read_kaggle_data(competition_name=""):
    """
    Generic function to read train and test input files into DataFrames 
    from Kaggle.
    
    Parameters
    ----------
    competition_name : string
        The string could be used to perform  operations that are
        specific for a Competition
    """
    if not skeleton_test:
        prepath = '../'
    else:
        prepath = ''
    input_folder = 'input/'
    filetype = '.csv'
    try:
        if competition_name == 'Porto Seguro':
            train = pd.read_csv(prepath + input_folder + 'train' + filetype)
            test = pd.read_csv(prepath + input_folder + 'test' + filetype)
        else:
            train = pd.read_csv(prepath + input_folder + 'train' + filetype)
            test = pd.read_csv(prepath + input_folder + 'test' + filetype)
    except FileNotFoundError:
        print("read_kaggle_data: Files not found")
        train = test = pd.DataFrame()
        
    return train, test

def reduce_memory_footprint(df):
    """
    Convert DataFrame columns from float64 to float32 and from int64 to int32.
    The operation reduces the memory footprint and speeds up numpy calculations
    """
    for col in df.columns:
        if df[col].dtypes == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtypes == 'int64':
            df[col] = df[col].astype('int32')
    
    return df


def timer(start_time=None):
    """
    Utility function to print the time taken to run a piece of code
    
    Parameters
    ----------
    start_time : datetime
        The first call must have no arguments, while the second call must have 
        the datetime returned by the second call.
    Example
    -------
    start = ff.timer()
        <code to be timed>
    ff.timer(start)
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def store_parms_results(competition_name, estimator_name, parms, results):
    """
    Utility function to keep historical records of models and results
    
    Parameters
    ----------
    competition_name : string
        The string could be used to perform  operations that are
        specific for a Competition
    estimator_name : string
        The string represents the estimator used for training. 
        For example 'xgboost', 'RandomForest', ...
    parms : dictionary
        Dictionary of parameters.
        For example {'eta': 0.05, 'max_depth': 4}
    results : DataFrame
        DataFrame representing the result of the training.
        For example???, 
    """