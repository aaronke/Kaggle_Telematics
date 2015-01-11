import cPickle as pickle
import numpy as np
import time
from sklearn.covariance import EmpiricalCovariance
from scipy.io import loadmat
from sklearn.preprocessing import Imputer

path = '/cshome/kzhou3/Data/feature/featureK/'
feature = loadmat(path + 'feature.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

output = open('/cshome/kzhou3/Dropbox/Telematics/submission/Out_42Feat.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = 2736
for k in range(1,size + 1):
    X = feature[200*(k-1):200*(k)]
    X = Imputer().fit_transform(X)
    emp_cov = EmpiricalCovariance().fit(X)
    mah = emp_cov.mahalanobis(X)
    MIN = mah.min()
    MAX = mah.max()
    predict = np.array([(MAX - i)/(MAX - MIN) for i in mah])
    for i in range(1,201):
        output.write(str(name[k - 1]) + '_' + str(i) + ',' + str(predict[i - 1]) + '\n')
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(size - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + str(k) + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
