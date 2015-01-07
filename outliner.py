import cPickle as pickle
import os
import numpy as np
import time
from sklearn.covariance import EmpiricalCovariance

directory = '/cshome/kzhou3/Data/feature/feature3/'
files = sorted(os.listdir(directory))
output = open('/cshome/kzhou3/Dropbox/Telematics/submission/Out_19Feat_1driver.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = len(files)
for f in files:
    X = []
    with open(directory + f,'rb') as fp:
        X += pickle.load(fp)
    X = np.array(X)
    emp_cov = EmpiricalCovariance().fit(X)
    mah = emp_cov.mahalanobis(X)
    MIN = mah.min()
    MAX = mah.max()
    predict = np.array([(MAX - i)/(MAX - MIN) for i in mah])
    for i in range(1,201):
        output.write(f + '_' + str(i) + ',' + str(predict[i - 1]) + '\n')
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(2736 - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + f + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
