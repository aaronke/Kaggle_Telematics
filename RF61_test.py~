import os
import time
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
from sklearn.preprocessing import Imputer
import cPickle as pickle
from sklearn import cross_validation as CV

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61_sort_en','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)


c = 0
size = 2736
for k in range(1,size + 1):
    X = feature[200*(k-1)+40:200*(k)]
    for i in range(1,2):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 200)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*160 + [1]*200)
    clf = RandomForest(n_estimators=250, max_features=0.2, max_depth=5, min_samples_split=2)
    #clf = GBRT(n_estimators=100, learning_rate=0.05, max_depth=4, max_features=5, min_samples_leaf=3, random_state=0, subsample = 0.5)
    scores = CV.cross_val_score(clf, X, y, cv=5)
    print "All test accuracy: " + str(scores)
    print "aveage is " + str(np.mean(np.array(scores)))
    X_test = feature[200*(k-1):200*(k)]
    y_test = np.array([0]*200)
    clf.fit(X, y)
    print "All accuracy" + str(clf.score(X_test, y_test))
    print "\n-------------------\n"
