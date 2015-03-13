import os
import time
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
import numpy as np
import cPickle as pickle

featureToUse = 'feature171'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + featureToUse + '_1','rb') as fp:
    f1 = np.array(pickle.load(fp))
with open(path + featureToUse + '_2','rb') as fp:
    f2 = np.array(pickle.load(fp))
with open(path + featureToUse + '_3','rb') as fp:
    f3 = np.array(pickle.load(fp))
with open(path + featureToUse + '_4','rb') as fp:
    f4 = np.array(pickle.load(fp))
feature = np.concatenate((f1,f2,f3,f4))
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF171/2V2_1888_estimator_8feat.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
neg = 200
for k in range(1,size + 1):
    X = feature[200*(k-1):200*k]
    for i in range(1,neg+1):
        X = np.concatenate((X, feature[(k -1 + i)%size*200:((k -1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X)
    y = np.array([0]*200 + [1]*neg)
    #clf = RandomForest(n_estimators=550, max_features=8, max_depth=None, min_samples_split=1) # 0.90080
    clf = RandomForest(n_estimators=1888, max_depth=None, min_samples_split=1) # 0.90464
    #clf = RandomForest(n_estimators=1888, max_features=8, max_depth=None, min_samples_split=1) # 0.90369
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
