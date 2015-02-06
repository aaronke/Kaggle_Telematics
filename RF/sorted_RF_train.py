import os
import time
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
import numpy as np
import cPickle as pickle

featureToUse = 'feature75'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + 'feature75','rb') as fp:
    feature = pickle.load(fp)
f = open('./ensemble/ensemble52_calib_nosort.csv', 'r')
f.readline()

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF75/75feat_sorted_200V200_1.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k)]
    for i in range(1,201):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*200 + [1]*200)
    clf = RandomForest(n_estimators=250, max_features=0.2, max_depth=5, min_samples_split=2)
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
