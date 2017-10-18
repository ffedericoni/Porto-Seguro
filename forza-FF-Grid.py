#https://www.kaggle.com/the1owl/forza-baseline/code
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
#import lightgbm as lgb

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
id_test = test['id'].values

col = [c for c in train.columns if c not in ['id','target']]
print(len(col))
col = [c for c in col if not c.startswith('ps_calc_')]
print(len(col))

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df

start_time = timer(None) # timing starts from this point for "start_time" variable

#params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
#          'objective': 'binary:logistic', 'eval_metric': 'auc', 
#          'scale_pos_weight': 1.25, 'random_state': 99, 'silent': True}
#x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.23, random_state=99)

#x1 = multi_transform(x1)
#x2 = multi_transform(x2)
#test = multi_transform(test)
print("Transforming...")
X = transform_df(train)
y = X['target']
test = transform_df(test)
print("Finished transformation...")

col = [c for c in X.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
print(X.values.shape, test.values.shape)
X = X[col]
test = test[col] #correzione
print(X.values.shape, test.values.shape)

#remove duplicates just in case
#tdups = multi_transform(train)
#tdups = transform_df(train)
#dups = tdups[tdups.duplicated(subset=col, keep=False)]
#
#x1 = x1[~(x1['id'].isin(dups['id'].values))]
#x2 = x2[~(x2['id'].isin(dups['id'].values))]
#print(x1.values.shape, x2.values.shape)

#y1 = x1['target']
#y2 = x2['target']
#x1 = x1[col]
#x2 = x2[col]

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1],
        'gamma': [1, 1.25],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': [4],
        'scale_pos_weight': [1.25]
        }
seed = 5
xgb = XGBClassifier(learning_rate=0.02, n_estimators=2000, objective='binary:logistic',
                    silent=True, missing=-1, random_state=seed)
folds = 5
param_comb = 12

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = seed)
sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.30, random_state=seed)

#random_search = RandomizedSearchCV(xgb, param_distributions=params, 
#                                   n_iter=param_comb, scoring='roc_auc', 
#                                   n_jobs=4, cv=sss.split(X,y), 
#                                   verbose=4, random_state=101
#                                   )
random_search = GridSearchCV(xgb, param_grid=params, 
                                   scoring='roc_auc', 
                                   n_jobs=4, cv=sss.split(X,y), 
                                   verbose=4
                                   )

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y
#                  , {
#       'early_stopping_rounds': 200,
#       'eval_metric': 'auc'
#       } 
        )
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


filename = 'Forza-xgb-d' + str(start_time.day) + '-h' + str(start_time.hour) + '.csv'
results_df.to_csv(filename, index=False)
#test[['id','target']].to_csv(filename, index=False, float_format='%.5f')
print("Filename=", filename)
#dumpfile = 'Forza-xgb-d' + str(start_time.day) + '-h' + str(start_time.hour) + '.dump'
#model.dump_model(dumpfile, fmap='', with_stats=True)


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
