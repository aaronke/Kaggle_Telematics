import numpy as np
from sklearn.preprocessing import StandardScaler as SS
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
import cPickle as pickle
import time
from sklearn.covariance import EmpiricalCovariance

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF_feat_weight_outliner.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
neg = 200
neg_step = 1
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k)]
    for i in range(1,neg + 1):
        X = np.concatenate((X, feature[(k-1 + i)%size*200:((k-1 + i)%size*200 + neg_step)]))
    X = Imputer().fit_transform(X)
    y = np.array([1.0]*200 + [0.0]*neg*neg_step)
    clf = RandomForest(n_estimators=250, max_features=8, max_depth=None, min_samples_split=1)
    clf.fit(X, y)
    weight = clf.feature_importances_
    scaler = SS().fit(X)
    X = scaler.transform(X)
    X = X*weight*100
    emp_cov = EmpiricalCovariance().fit(X)
    mah = emp_cov.mahalanobis(X)
    MIN = mah.min()
    MAX = mah.max()
    scores = np.array([(MAX - i)/(MAX - MIN) for i in mah])
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