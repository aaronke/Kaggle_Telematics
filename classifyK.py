import os
import time
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.preprocessing import Imputer
import numpy as np
from scipy.io import loadmat
import cPickle as pickle

# ============= load feature & name data ==================
path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

# ============= initial output prediction file ==================
output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF_2V2.csv', 'w')
output.write('driver_trip,prob\n')

# ============= start timing and trining ==================
start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k)] # positive training data
    # add other 200 negative training data
    for i in range(1,201):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + 1)]))
    X = Imputer().fit_transform(X) # imputer for missing values
    y = np.array([0]*200 + [1]*200)
    clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1) # set classifier parameters
    clf.fit(X, y) # train the classifier
    scores = clf.predict_proba(X[:200])[:,0] # use the classifier to predict
    # ============= write prediction into the output file ==================
    for i in range(1,201):
        output.write(str(name[k - 1]) + '_' + str(i) + ',' + str(scores[i - 1]) + '\n')
    # ============= update timing ==================
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(size - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + str(k) + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
