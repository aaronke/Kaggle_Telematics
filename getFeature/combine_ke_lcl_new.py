import cPickle as pickle
import os
import time
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import Imputer

path = '/cshome/kzhou3/Data/feature/feature_ke_lcl_new/'
feature = loadmat(path + 'feature_34.mat')
name = loadmat(path + 'name.mat')
feature = feature['drivers_features'][:547200]
name = name['Sort_Names'][0]

fff = np.array([[0]*134])

start = time.time()
c = 0
for k in range(0,2736):
    index = name[k]
    lcl = open('/cshome/kzhou3/Data/feature/feature_ke_lcl_new/Train/' + str(index) + '.txt','rb')
    f_l = np.array([[0]*100])
    for line in lcl:
        one_f_l = [float(i) for i in line.strip().split(' ')]
        f_l = np.concatenate((f_l,np.array([one_f_l])))
    f_l = f_l[1:]
    f_k = feature[200*(k):200*(k+1)]
    X = np.concatenate((f_k, f_l), axis=1)
    fff = np.concatenate((fff, X))
    print "DONE" + str(k)
end = time.time()
fff = fff[1:]
f1 = fff[:1368*100]
f2 = fff[1368*100:1368*200]
f3 = fff[1368*200:1368*300]
f4 = fff[1368*300:]
with open('/cshome/kzhou3/Data/feature/feature134/feature134_1','wb') as fp:
    pickle.dump(f1,fp)
with open('/cshome/kzhou3/Data/feature/feature134/feature134_2','wb') as fp:
    pickle.dump(f2,fp)
with open('/cshome/kzhou3/Data/feature/feature134/feature134_3','wb') as fp:
    pickle.dump(f3,fp)
with open('/cshome/kzhou3/Data/feature/feature134/feature134_4','wb') as fp:
    pickle.dump(f4,fp)
print "=====Total time: " + str((end - start)/60) + " minutes ======"
