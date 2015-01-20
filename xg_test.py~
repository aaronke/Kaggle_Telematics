#!/usr/bin/python
import sys
import numpy as np
# append the path to xgboost, you may need to change the following line
# alternatively, you can add the path to PYTHONPATH environment variable
sys.path.append('/cshome/kzhou3/xgboost/wrapper')
import xgboost as xgb
import cPickle as pickle
from sklearn.preprocessing import Imputer

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61_sort','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    return (dtrain, dtest, param)

size = 2736
neg = 500
neg_step = 1
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k)]
    for i in range(1,neg + 1):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + neg_step)]))
    X = Imputer().fit_transform(X)
    y = np.array([1.0]*200 + [0.0]*neg*neg_step)
    weight = np.array([2.0]*100 + [3.0]*100 + [1.0]*(len(y)-200))
    # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
    xgmat = xgb.DMatrix(X, label=y)
    sum_wpos = sum( weight[i] for i in range(len(y)) if y[i] == 1.0  )
    sum_wneg = sum( weight[i] for i in range(len(y)) if y[i] == 0.0  )
    param = {'objective':'binary:logistic', 'eval_metric':'error', 'scale_pos_weight':sum_wneg/sum_wpos, 'max_depth':6, 'eta':0.05, 'silent':1,'bst:min_child_weight':8,'bst:gamma':5,'bst:subsample':0.5}
#param = {'objective':'binary:logistic', 'max_depth':5, 'eta':0.05, 'silent':1,'bst:min_child_weight':5,'bst:gamma':3,'bst:subsample':0.5}
    num_round = 250
    xgb.cv(param, xgmat, num_round, nfold=5, metrics={'error','auc'}, seed = 0, fpreproc = fpreproc)
