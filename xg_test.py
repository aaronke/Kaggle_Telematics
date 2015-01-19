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
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

k = 1 # choose which driver to classify
X = feature[200*(k-1):200*(k)]
neg = 800
neg_step = 4
for i in range(1,neg + 1):
    X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + neg_step)]))
X = Imputer().fit_transform(X)
y = np.array([1.0]*200 + [0.0]*neg*neg_step)
weight = np.array([2.0]*200 + [1.0]*(len(y)-200))


print ('loading numpy feature end')

############ xgboost ##############
# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix(X, label=y)
sum_wpos = sum( weight[i] for i in range(len(y)) if y[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(y)) if y[i] == 0.0  )
# setup parameters for xgboost
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
# scale weight of positive examples
param = {'objective':'binary:logistic', 'eval_metric':'auc', 'scale_pos_weight':sum_wneg/sum_wpos, 'max_depth':6, 'eta':0.01, 'silent':1,'bst:min_child_weight':20,'bst:gamma':10,'bst:subsample':0.5}
#param = {'objective':'binary:logistic', 'max_depth':5, 'eta':0.05, 'silent':1,'bst:min_child_weight':5,'bst:gamma':3,'bst:subsample':0.5}

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    return (dtrain, dtest, param)

# boost 120 trees
num_round = 600
# you can directly throw param in, though we want to watch multiple metrics here 

watchlist = [ (xgmat,'train') ]
print ('loading data end, start to boost trees')
bst = xgb.train( param, xgmat, num_round, watchlist )
xgb.cv(param, xgmat, num_round, nfold=5, metrics={'error','auc'}, seed = 0, fpreproc = fpreproc)
