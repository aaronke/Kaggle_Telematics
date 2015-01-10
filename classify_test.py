import cPickle as pickle
from sklearn.svm import NuSVC
from sklearn import cross_validation as CV
from sklearn.metrics import roc_auc_score as ROC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
import os
path = '/cshome/kzhou3/Data/feature/feature3/'
files = sorted(os.listdir(path))
with open(path + '1','rb') as fp:
	feature1 = pickle.load(fp)
with open(path + '10','rb') as fp:
	feature2 = pickle.load(fp)
with open(path + '100','rb') as fp:
	feature3 = pickle.load(fp)
with open(path + '1000','rb') as fp:
	feature4 = pickle.load(fp)
with open(path + '1001','rb') as fp:
	feature5 = pickle.load(fp)
with open(path + '1002','rb') as fp:
	feature6 = pickle.load(fp)
with open(path + '1003','rb') as fp:
	feature7 = pickle.load(fp)
with open(path + '1004','rb') as fp:
	feature8 = pickle.load(fp)
with open(path + '1005','rb') as fp:
	feature9 = pickle.load(fp)
with open(path + '1006','rb') as fp:
	feature10 = pickle.load(fp)
with open(path + '1007','rb') as fp:
	feature11 = pickle.load(fp)
#X = feature1 + feature2[:50] + feature3[:50] + feature4[:50] + feature5[:50]
#X = feature1 + feature3
#X = feature1 + feature2[:100] + feature3[:100] + feature4[:100] + feature5[:100] + feature6[:100] + feature7[:100] + feature8[:100] + feature9[:100] + feature10[:100] + feature11[:100]
#y = [0]*200 + [1]*1000
X = []
y = [0]*200 + [1]*800
f = 0
with open(path + files[f], 'rb') as fp:
    X += pickle.load(fp)
for i in range(1,101):
    with open(path + files[f + i], 'rb') as fp:
        X += pickle.load(fp)[:8]
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
#clf = NuSVC().fit(X_train, y_train)

#clf = RandomForest(n_estimators=250, max_features=4, max_depth=None, min_samples_split=1)
clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6)
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
scores = CV.cross_val_score(clf, X, y, cv=5)
print "All test accuracy: " + str(scores)
print "aveage is " + str(np.mean(np.array(scores)))

clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
print clf.score(X_train, y_train)
y_score = clf.predict_proba(X)[:, 1]
print ROC(y, y_score)
y_score = clf.predict_proba(X_test)[:, 0]
print y_score
print clf.predict(X_test)
"""
print clf.classes_
#y_score = clf.predict_proba(X[:50])[:, 1]
#print y_score
#print clf.predict_proba(X)[0][0]

X = X_train[:150] + X_test[:50]
predict = clf.predict_proba(X)[:,0]
predictS = np.sort(predict)
flag = predictS[100]
X_new = []
for i in range(0,200):
    if predict[i] >= flag:
        X_new.append(X[i])
X = X_new[:100]
y = [0]*100 + [1]*1000
f = 213
for i in range(1,101):
    with open(path + files[f + i], 'rb') as fp:
        X += pickle.load(fp)[:10]
clf.fit(X, y)
print clf.score(X_test, y_test)
print clf.score(X_train, y_train)
y_score = clf.predict_proba(X)[:, 1]
print ROC(y, y_score)
y_score = clf.predict_proba(X_test)[:, 0]
print y_score
"""
