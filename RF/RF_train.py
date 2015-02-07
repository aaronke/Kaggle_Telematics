import os
import time
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
import numpy as np
import cPickle as pickle

featureToUse = 'feature75'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + featureToUse,'rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF75/4V4_550estimator_std.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
neg = 400
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k+1)]
    for i in range(1,neg + 1):
        X = np.concatenate((X, feature[(k + i)%size*200:((k + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*400 + [1]*neg)
    # clf = RandomForest(n_estimators=250, max_features=0.2, max_depth=5, min_samples_split=2) # 0.837
    # clf = RandomForest(n_estimators=450, max_features=0.25, max_depth=5, min_samples_split=2)
    clf = RandomForest(n_estimators=550, max_features=8, max_depth=None, min_samples_split=1)
    # clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
    clf.fit(X, y)
    # ======== Predict ===============
    X = feature[200*(k-1):200*(k)]
    X = Imputer().fit_transform(X)
    scores = clf.predict_proba(X[:200])[:,0]
    #======== get right driver lable ===
    for i in range(1, 201):
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
