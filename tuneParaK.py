from sklearn import cross_validation as CV
from sklearn.metrics import roc_auc_score as ROC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
import numpy as np
from scipy.io import loadmat
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

param_grid_svm = {'C': [1, 0.1, 0.05, 0.001, 0.0005],'degree':[2,3,4,5,6]}
param_grid = {'learning_rate': [0.2, 0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 5, 6, 7, 8, 9],
              'min_samples_leaf': [1, 3, 5, 9, 17],
              'max_features': [1.0, 0.5, 0.3, 0.1],
              'subsample': [0.3, 0.4, 0.5, 0.6, 0.7]
              }

k = 31 # choose which driver to classify
X = feature[200*(k-1):200*(k)]
for i in range(1,301):
    X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + 3)]))

y = np.array([0]*200 + [1]*900)

clf = SVC(kernel='poly', shrinking=True, probability=True)
#clf = GBRT(n_estimators=250)
gs_cv = GridSearchCV(clf, param_grid_svm, n_jobs=4).fit(X, y)
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
    scores = CV.cross_val_score(clf, X, y, cv=5)
    print "All test accuracy: " + str(scores)
    print "aveage is " + str(np.mean(np.array(scores)))
    clf.fit(X_train, y_train)
    print "Test accuracy: " + str(clf.score(X_test, y_test))
    print "Train accuracy: "+ str(clf.score(X_train, y_train))
    y_score = clf.predict_proba(X)[:, 1]
    print "ROC area: " + str(ROC(y, y_score))
"""
