from sklearn import cross_validation as CV
from sklearn.metrics import roc_auc_score as ROC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

k = 17 # choose which driver to classify
X = feature[200*(k-1):200*(k)]
for i in range(1,201):
    X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + 1)]))

y = np.array([0]*200 + [1]*200)

X = Imputer().fit_transform(X)

X_train = np.concatenate((X[:150], X[250:]))
y_train =  np.concatenate((y[:150], y[250:]))
X_test = X[150:250]
y_test = y[150:250]

clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
#clf = SVC(C=0.0005, kernel='poly', degree=5, gamma=0.0, coef0=0.0, shrinking=True, probability=True)
#clf = GBRT(n_estimators=250, learning_rate=0.05, max_depth=8, max_features=1.0, min_samples_leaf=17,  random_state=0, subsample=0.5)
#clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6)
#clf = Pipeline([("scale", StandardScaler()), ("gbrt", GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6))])

scores = CV.cross_val_score(clf, X, y, cv=5)
print "All test accuracy: " + str(scores)
print "aveage is " + str(np.mean(np.array(scores)))

clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
print clf.score(X_train, y_train)
print "=====================\n"

k = 27 # choose which driver to classify
X = feature[200*(k-1):200*(k)]
for i in range(1,201):
    X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + 1)]))
X = Imputer().fit_transform(X)
X_train = np.concatenate((X[:150], X[250:]))
X_test = X[150:250]
clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
#clf = SVC(C=0.0005, kernel='poly', degree=5, gamma=0.0, coef0=0.0, shrinking=True, probability=True)
#clf = GBRT(n_estimators=250, learning_rate=0.05, max_depth=8, max_features=1.0, min_samples_leaf=17,  random_state=0, subsample=0.5)
#clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6)
#clf = Pipeline([("scale", StandardScaler()), ("gbrt", GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6))])
scores = CV.cross_val_score(clf, X, y, cv=5)
print "All test accuracy: " + str(scores)
print "aveage is " + str(np.mean(np.array(scores)))

clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
print clf.score(X_train, y_train)
print "=====================\n"

k = 157 # choose which driver to classify
X = feature[200*(k-1):200*(k)]
for i in range(1,201):
    X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + 1)]))
X = Imputer().fit_transform(X)
X_train = np.concatenate((X[:150], X[250:]))
X_test = X[150:250]
clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
#clf = SVC(C=0.0005, kernel='poly', degree=5, gamma=0.0, coef0=0.0, shrinking=True, probability=True)
#clf = GBRT(n_estimators=250, learning_rate=0.05, max_depth=8, max_features=1.0, min_samples_leaf=17,  random_state=0, subsample=0.5)
#clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6)
#clf = Pipeline([("scale", StandardScaler()), ("gbrt", GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6))])
scores = CV.cross_val_score(clf, X, y, cv=5)
print "All test accuracy: " + str(scores)
print "aveage is " + str(np.mean(np.array(scores)))

clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
print clf.score(X_train, y_train)
print "=====================\n"

"""
y_score = clf.predict_proba(X)[:, 1]
print ROC(y, y_score)
print y_score[:50]
y_score = clf.predict_proba(X_test)[:, 0]
print y_score
#print clf.predict(X_test)
#print clf.predict(X[:50])
"""