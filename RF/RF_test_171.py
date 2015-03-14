import os
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from sklearn.preprocessing import Imputer
import cPickle as pickle
from sklearn import cross_validation as CV

featureToUse = 'feature171'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + featureToUse + '_1','rb') as fp:
    f1 = np.array(pickle.load(fp))
with open(path + featureToUse + '_2','rb') as fp:
    f2 = np.array(pickle.load(fp))
with open(path + featureToUse + '_3','rb') as fp:
    f3 = np.array(pickle.load(fp))
with open(path + featureToUse + '_4','rb') as fp:
    f4 = np.array(pickle.load(fp))
feature = np.concatenate((f1,f2,f3,f4))
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
        X = np.concatenate((X, feature[(k -1 + i)%size*200:((k -1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*200 + [1]*neg)
    #clf = RandomForest(n_estimators=250, max_features=0.2, max_depth=5, min_samples_split=2) # 0.837
    #clf = RandomForest(n_estimators=1000, max_features=13, max_depth=None, min_samples_split=1)
    #clf = RandomForest(n_estimators=1000, max_features=8, max_depth=None, min_samples_split=1)
    #clf = GBRT(n_estimators=550, learning_rate=0.05, max_depth=6, max_features="auto", min_samples_leaf=5, random_state=0, subsample = 0.5)
    clf = LR(penalty='l2', dual=False, C=0.8, fit_intercept=True, intercept_scaling=1, random_state=None)
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
