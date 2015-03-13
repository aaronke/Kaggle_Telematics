import os
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import GradientBoostingClassifier as GBRT
import numpy as np
from sklearn.preprocessing import Imputer
import cPickle as pickle
from sklearn import cross_validation as CV

featureToUse = 'feature75'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + featureToUse,'rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

c = 0
size = 2736
all_ave = []
all_pred = []
neg = 200
for k in range(1,size + 1):
    X = feature[200*(k-1):200*k]
    for i in range(1,neg+1):
        X = np.concatenate((X, feature[(k - 1 + i)%size*200:((k - 1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*200 + [1]*neg)
    # clf = RandomForest(n_estimators=250, max_features=0.2, max_depth=5, min_samples_split=2) # 0.837
    clf = RandomForest(n_estimators=550, max_features=8, max_depth=None, min_samples_split=1)
    # clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.2, min_samples_leaf=3, random_state=0, subsample = 0.6)
    scores = CV.cross_val_score(clf, X, y, cv=5)
    print "aveage cv score for this driver " + str(np.mean(np.array(scores)))
    all_ave.append(np.mean(np.array(scores)))
    print "aveage cv score for all drives" + str(reduce(lambda x, y: x + y, all_ave) / len(all_ave))
    X_test = feature[200*(k-1):200*(k)]
    y_test = np.array([0]*200)
    clf.fit(X, y)
    print "All accuracy" + str(clf.score(X_test, y_test))
    all_pred.append(clf.score(X_test, y_test))
    print "aveage cv score for all drives" + str(reduce(lambda x, y: x + y, all_pred) / len(all_pred))
    print "\n-------------------\n"
