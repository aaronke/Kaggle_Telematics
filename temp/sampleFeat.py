import cPickle as pickle
import os
import time
from sklearn.svm import NuSVC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import Imputer

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
sample = feature[:10000]

with open('/cshome/kzhou3/Data/feature/feature61/feature61_top50','wb') as fp:
    pickle.dump(sample,fp)
