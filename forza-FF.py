#https://www.kaggle.com/the1owl/forza-baseline/code
"""This is the forza-FF.py source code"""
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
import xgboost as xgb
#import lightgbm as lgb
from datetime import datetime




def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#ff drop least important features
#worst 5, excluding ps_ind_14 = ps_ind_10_bin + ps_ind_11_bin + ps_ind_12_bin + ps_ind_13_bin
li_features = ['ps_ind_13_bin', 'ps_ind_10_bin',
'ps_ind_11_bin', 'ps_ind_12_bin']
#second worst 5
#li_features += ['ps_car_10_cat', 'ps_calc_20_bin',
#'ps_ind_18_bin', 'ps_calc_15_bin', 'ps_calc_16_bin']
#li_features = [] #Dont drop
train = train.drop(li_features, axis=1)
test = test.drop(li_features, axis=1)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

col = [c for c in train.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}
#%%
def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
#        print('Now transforming col ', c)
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            #df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)

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

train = reduce_memory_footprint(train)
test = reduce_memory_footprint(test)

start_time = timer(None) # timing starts from this point for "start_time" variable

seed=77
params = {'eta': 0.05, 'max_depth': 4, 'subsample': 0.7, 'colsample_bytree': 0.8,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 
          'gamma': 4, 'random_state': seed, 'silent': True,
          'min_child_weight': 10, 'nthread': 3,
          'max_delta_step':5}
test_perc = 0.25
x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], 
                                                  test_size=test_perc, 
                                                  random_state=seed)

#unc x1 = transform_df(x1)
#unc x2 = transform_df(x2)
test = transform_df(test)

col = [c for c in test.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
#print(x1.values.shape, x2.values.shape)

#remove duplicates just in case
#tdups = multi_transform(train)
tdups = transform_df(train)
print('Train Shape: ', tdups[col].shape)
#dups = tdups[tdups.duplicated(subset=col, keep=False)]


#unc x1 = x1[~(x1['id'].isin(dups['id'].values))]
#unc x2 = x2[~(x2['id'].isin(dups['id'].values))]
#print(x1.values.shape, x2.values.shape)

#unc y1 = x1['target']
#unc y2 = x2['target']
#unc x1 = x1[col]
#unc x2 = x2[col]
#%%
#unc watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
#unc model = xgb.train(params, xgb.DMatrix(x1, y1), 2500,  watchlist, #feval=gini_xgb, 
#unc                   maximize=True, verbose_eval=100, early_stopping_rounds=200)
ntrees = 800
ev_hist = xgb.cv(params, xgb.DMatrix(tdups[col], tdups['target']), ntrees,  nfold=5, 
                  metrics={'auc'}, seed=seed, stratified=True,
                  maximize=True, verbose_eval=int(ntrees/10), as_pandas = True)
timer(start_time)


raise NameError('End of Run')



test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
#test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
timer(start_time) # timing ends here for "start_time" variable
filename = 'Forza-xgb-d' + str(start_time.day) + '-h' + str(start_time.hour) + '.csv'
test[['id','target']].to_csv(filename, index=False, float_format='%.5f')
print("Filename=", filename)
print("Params=", params)
print("Test % =", test_perc, "Seed =", seed)
dumpfile = 'Forza-xgb-d' + str(start_time.day) + '-h' + str(start_time.hour) + '.dump'
model.dump_model(dumpfile, fmap='', with_stats=False)

print(datetime.now(), '\a')
print(__doc__)

##LightGBM
#def gini_lgb(preds, dtrain):
#    y = list(dtrain.get_label())
#    score = gini(y, preds) / gini(y, y)
#    return 'gini', score, True
#
#params = {'learning_rate': 0.02, 'max_depth': 4, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'is_training_metric': False, 'seed': 99}
#model2 = lgb.train(params, lgb.Dataset(x1, label=y1), 1000, lgb.Dataset(x2, label=y2), verbose_eval=50, feval=gini_lgb, early_stopping_rounds=200)
#test['target'] = model2.predict(test[col], num_iteration=model2.best_iteration)
#test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
#test[['id','target']].to_csv('lgb_submission.csv', index=False, float_format='%.5f')
#
#df1 = pd.read_csv('xgb_submission.csv')
#df2 = pd.read_csv('lgb_submission.csv')
#df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
#blend = pd.merge(df1, df2, how='left', on='id')
#for c in df1.columns:
#    if c != 'id':
#        blend[c] = (blend[c] * 0.5)  + (blend[c+'_'] * 0.5)
#blend = blend[df1.columns]
#blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0,1)
#blend.to_csv('blend1.csv', index=False, float_format='%.5f')
