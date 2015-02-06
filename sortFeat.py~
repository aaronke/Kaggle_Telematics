import os
import time
import numpy as np
import cPickle as pickle

path = '/cshome/kzhou3/Data/feature/feature75/'
with open(path + 'feature75','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

f = open('./ensemble/ensemble52_calib_nosort.csv', 'r')
f.readline()

feature_sort = np.array([[0]*75])
for k in range(0, 2736):
    X_new = []
    for j in range(0,200):
        line = f.readline().split(',') # 'driverID_tripID, rank'
        namei = line[0].split('_') # 'driverID_tripID'
        kk = int(np.where(name==int(namei[0]))[0])
        X_new.append(feature[kk*200 + int(namei[1]) - 1])
    feature_sort = np.concatenate((feature_sort, np.array(X_new)))
    if k%20 == 0:
        print "DONE" + str(k)

feature_sort = feature_sort[1:]
print feature_sort.shape
with open('/cshome/kzhou3/Data/feature/feature75/feature75_sort','wb') as fp:
    pickle.dump(feature_sort,fp)
f.close()
