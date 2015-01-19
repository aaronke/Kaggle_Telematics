import cPickle as pickle
from sklearn import cross_validation as CV
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
import os
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer

param_grid = {'max_depth': [None, 4, 5, 6, 7, 8],
              'min_samples_leaf': [1, 2, 3, 5, 9],
              'max_features': [0.8, 0.5, 0.3, 0.1, "auto", "log2"],
              }

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)
for k in range(1,11):
    print "\n-------------------------------------------------------\n"
    X = feature[200*(k-1):200*(k)]
    for i in range(1,201):
        X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + 1)]))
    y = np.array([0]*200 + [1]*200)
    X = Imputer().fit_transform(X)
    clf = RandomForest(n_estimators=250)
    gs_cv = GridSearchCV(clf, param_grid, n_jobs=6).fit(X, y)
    print gs_cv.best_params_
    """
    kf = CV.KFold(n_elements, n_folds = 5, indices = False)
    for train, test in kf:
        clf.fit(X[train], y[train])
        accu_test = clf.score(X[test], y[test])
        accu_train = clf.score(X[train], y[train])
        y_score = clf.predict_proba(X)[:, 1]
        roc = ROC(y, y_score)
        print "Test accuracy: " + str(accu_test) + ". Train accuracy: " + str(accu_train) + ". ROC area: " + str(roc)
    """
    """
    scores = CV.cross_val_score(clf, X, y, cv=5)
    print "All test accuracy: " + str(scores)
    print "aveage is " + str(np.mean(np.array(scores)))
    clf.fit(X_train, y_train)
    print "Test accuracy: " + str(clf.score(X_test, y_test))
    print "Train accuracy: "+ str(clf.score(X_train, y_train))
    y_score = clf.predict_proba(X)[:, 1]
    print "ROC area: " + str(ROC(y, y_score))
    """
    print "\n-------------------------------------------------------\n"
