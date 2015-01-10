import os
import time
from sklearn.ensemble import GradientBoostingClassifier as GBRT
import numpy as np
from scipy.io import loadmat

path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/GBRT_Ke_42.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k)]
    for i in range(1,101):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 8)]))
    y = np.array([0]*200 + [1]*800)
    clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3, random_state=0, subsample = 0.6)
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
