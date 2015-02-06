import os
import time
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
import numpy as np
import cPickle as pickle

featureToUse = 'feature75'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + 'feature75' + '_sort','rb') as fp:
    feature = pickle.load(fp)
f = open('/cshome/kzhou3/Dropbox/Telematics/ensemble/ensemble52_calib_nosort.csv', 'r')
f.readline()

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF75/sorted165V165.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    top_offset = 35
    X = feature[200*(k-1)+top_offset:200*(k)]
    for i in range(1,201-top_offset):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*(200-top_offset) + [1]*(200-top_offset))
    clf = RandomForest(n_estimators=250, max_features=0.2, max_depth=5, min_samples_split=2)
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
