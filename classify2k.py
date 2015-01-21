import os
import time
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
import numpy as np
import cPickle as pickle
path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61_sort_en','rb') as fp:
    feature = pickle.load(fp)
f = open('./ensemble/ensemble52_calib_nosort.csv', 'r')
f.readline()

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF_2v2_sorted.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    X = feature[200*(k-1)+40:200*(k)]
    for i in range(1,2):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 200)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*160 + [1]*200)
    #clf = SVC(C=0.0005, kernel='poly', degree=5, gamma=0.0, coef0=0.0, shrinking=True, probability=True)
    clf = RandomForest(n_estimators=250, max_features=5, max_depth=None, min_samples_split=1)
    #clf = GBRT(n_estimators=250, learning_rate=0.05, max_depth=8, max_features=1.0, min_samples_leaf=17, random_state=0, subsample = 0.5)
    clf.fit(X, y)
    # ======== Predict ===============
    X = feature[200*(k-1):200*(k)]
    X = Imputer().fit_transform(X)
    scores = clf.predict_proba(X[:200])[:,0]
    #======== get right driver lable ===
    for i in range(1, 201):
        line = f.readline().split(',') # 'driverID_tripID, rank'
        output.write(line[0] + ',' + str(scores[i - 1]) + '\n')
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(size - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + str(k) + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
