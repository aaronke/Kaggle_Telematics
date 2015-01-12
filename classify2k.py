import os
import time
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np
from sklearn.preprocessing import Imputer
import cPickle as pickle

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF_61_2V2.csv', 'w')
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
    #clf = SVC(C=0.0005, kernel='poly', degree=5, gamma=0.0, coef0=0.0, shrinking=True, probability=True)
    clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
    #clf = GBRT(n_estimators=250, learning_rate=0.05, max_depth=8, max_features=1.0, min_samples_leaf=17, random_state=0, subsample = 0.5)
    clf.fit(X, y)
    scores = clf.predict_proba(X[:200])[:,0]
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
