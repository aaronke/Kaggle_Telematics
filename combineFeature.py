import cPickle as pickle
import os
import time
from sklearn.svm import NuSVC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import Imputer

path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

directory = '/cshome/kzhou3/Data/feature/feature3/'
files = sorted(os.listdir(directory))
fff = np.array([[0]*61])

start = time.time()
c = 0
size = len(files)
for k in range(0,2736):
    index = name[k]
    with open(directory + str(index),'rb') as fp:
        X = np.array(pickle.load(fp))
    X2 = feature[200*(k):200*(k+1)]
    X = np.concatenate((X, X2), axis=1)
    fff = np.concatenate((fff, X))
    print "DONE" + str(k)
end = time.time()
fff = fff[1:]
kkk = name
with open('/cshome/kzhou3/Data/feature/feature61/feature61','wb') as fp:
    pickle.dump(fff,fp)
with open('/cshome/kzhou3/Data/feature/feature61/name','wb') as fp:
    pickle.dump(kkk,fp)
print "=====Total time: " + str((end - start)/60) + " minutes ======"
