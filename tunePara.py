import cPickle as pickle
from sklearn import cross_validation as CV
from sklearn.metrics import roc_auc_score as ROC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
import numpy as np
import os
from sklearn.grid_search import GridSearchCV

param_grid = {'learning_rate': [0.2, 0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 5, 6],
              'min_samples_leaf': [1, 3, 5, 9, 17],
              'max_features': [1.0, 0.3, 0.1],
              'subsample': [0.4, 0.5, 0.6]
              }

path = '/cshome/kzhou3/Data/feature/feature3nor/'
files = sorted(os.listdir(path))
for ii in range(1,11):
    print "\n-------------------------------------------------------\n"
    X = []
    y = [0]*200 + [1]*200
    f = ii
    with open(path + files[f], 'rb') as fp:
        X += pickle.load(fp)
    for i in range(1,101):
        with open(path + files[f + i], 'rb') as fp:
            X += pickle.load(fp)[:2]
    
    X_train = X[:150] + X[250:]
    y_train = y[:150] + y[250:]
    X_test = X[150:250]
    y_test = y[150:250]
    X = np.array(X)
    n_elements = X.shape[0]
    y = np.array(y)
    c = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
    np.random.shuffle(c)
    X = c[:, :X.size//len(X)].reshape(X.shape)
    y = c[:, X.size//len(X):].reshape(y.shape)
    
#    clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=5, random_state=0, subsample=0.5)
    clf = GBRT(n_estimators=250)
    gs_cv = GridSearchCV(clf, param_grid, n_jobs=4).fit(X, y)
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
