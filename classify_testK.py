from sklearn import cross_validation as CV
from sklearn.metrics import roc_auc_score as ROC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
import numpy as np
from scipy.io import loadmat

path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

k = 31 # choose which driver to classify
X = feature[200*(k-1):200*(k)]
for i in range(1,101):
    X = np.concatenate((X, feature[(k-1 + i)%2736*200:((k-1 + i)%2736*200 + 8)]))

y = np.array([0]*200 + [1]*800)


X_train = np.concatenate((X[:150], X[250:]))
y_train =  np.concatenate((y[:150], y[250:]))
X_test = X[150:250]
y_test = y[150:250]

#clf = RandomForest(n_estimators=250, max_features=4, max_depth=None, min_samples_split=1)
clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3,  random_state=0, subsample=0.6)

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
