import os
import time
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
from scipy.io import loadmat
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import Imputer
from sklearn import cross_validation as CV
import cPickle as pickle

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF_61_twoRound.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    # Round 1: get top 140 relavent trips
    X_predict = feature[200*(k-1):200*(k)]
    X_predict = Imputer().fit_transform(X_predict)
    X = feature[200*(k-1):200*(k)]
    X = Imputer().fit_transform(X)
    emp_cov = EmpiricalCovariance().fit(X)
    mah = emp_cov.mahalanobis(X)
    MIN = mah.min()
    MAX = mah.max()
    scores = np.array([(MAX - i)/(MAX - MIN) for i in mah])
    scoresII = np.sort(scores)
    # only use the top 140 as training data	
    flag = scoresII[60]
    X_new = []
    for i in range(0,200):
        if scores[i] >= flag:
            X_new.append(X[i])
    # Round 2: train the rest
    X = np.array(X_new[:140])
    for i in range(1,201):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*140 + [1]*200)
    clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
    #clf = GBRT(n_estimators=250, learning_rate=0.05, max_depth=8, max_features=1.0, min_samples_leaf=17, random_state=0, subsample = 0.5)
    clf.fit(X, y)
    scores = clf.predict_proba(X_predict)[:,0]
    #print sum(clf.predict(X_predict))
    for i in range(1,201):
        output.write(str(name[k - 1]) + '_' + str(i) + ',' + str(scores[i - 1]) + '\n')
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(size - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + str(k) + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
