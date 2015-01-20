#!/usr/bin/python
import sys
import numpy as np
# append the path to xgboost, you may need to change the following line
# alternatively, you can add the path to PYTHONPATH environment variable
sys.path.append('/cshome/kzhou3/xgboost/wrapper')
import xgboost as xgb
import os
import time
from sklearn.preprocessing import Imputer
import cPickle as pickle
from scipy.io import loadmat

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'feature61_half','rb') as fp:
    feature_half = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/XG_half_all_1V4.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
neg = 400
neg_step = 1
for k in range(1,size + 1):
    X = feature_half[100*(k-1):(100*k)]
    for i in range(1,neg + 1):
        X = np.concatenate((X, feature_half[(k-1 + i)%size*100:((k-1 + i)%size*100 + neg_step)]))
    X = Imputer().fit_transform(X)
    y = np.array([1.0]*100 + [0.0]*neg*neg_step)
    weight = np.array([2.0]*100 + [1.0]*(len(y)-100))
    # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
    xgmat = xgb.DMatrix(X, label=y)
    sum_wpos = sum( weight[i] for i in range(len(y)) if y[i] == 1.0  )
    sum_wneg = sum( weight[i] for i in range(len(y)) if y[i] == 0.0  )
    # ==== tune xgboosting param and train =======
    #param = {'objective':'binary:logistic', 'max_depth':5, 'eta':0.05, 'silent':1,'bst:min_child_weight':5,'bst:gamma':3,'bst:subsample':0.5} # 0.78
    #param = {'objective':'binary:logistic', 'eval_metric':'auc', 'scale_pos_weight':sum_wneg/sum_wpos, 'max_depth':6, 'eta':0.01, 'silent':1,'bst:min_child_weight':20,'bst:gamma':10,'bst:subsample':0.5} # 0.76
    param = {'objective':'binary:logistic', 'eval_metric':'error', 'scale_pos_weight':sum_wneg/sum_wpos, 'max_depth':4, 'eta':0.01, 'silent':1,'bst:min_child_weight':10,'bst:gamma':5,'bst:subsample':0.5}
    num_round = 100
    watchlist = [(xgmat,'train')]
    bst = xgb.train( param, xgmat, num_round, watchlist )
    # ======== Predict ===============
    X = feature[200*(k-1):200*(k)]
    X = Imputer().fit_transform(X)
    y = np.array([1.0]*200)
    xgmat = xgb.DMatrix(X, label=y)
    scores = bst.predict(xgmat)
    # write predict to file
    for i in range(1,201):
        output.write(str(name[k - 1]) + '_' + str(i) + ',' + str(scores[i - 1]) + '\n')
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(size - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + str(k) + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
