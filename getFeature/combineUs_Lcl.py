import cPickle as pickle
import os
import time
import numpy as np

featureToUse = 'feature75'
path = '/cshome/kzhou3/Data/feature/' + featureToUse + '/'
with open(path + featureToUse,'rb') as fp:
    feature = pickle.load(fp)
with open(path + 'name','rb') as fp:
    name = pickle.load(fp)

fff = np.array([[0]*171])
start = time.time()
for k in range(0,2736):
    index = name[k]
    with open('/cshome/kzhou3/Data/feature/feature96_lcl/' + str(index),'rb') as fp:
        X = np.array(pickle.load(fp))
    X2 = feature[200*(k):200*(k+1)]
    X = np.concatenate((X, X2), axis=1)
    fff = np.concatenate((fff, X))
    print "DONE" + str(k)
end = time.time()
fff = fff[1:]
f1 = fff[:1368*100]
f2 = fff[1368*100:1368*200]
f3 = fff[1368*200:1368*300]
f4 = fff[1368*300:]
with open('/cshome/kzhou3/Data/feature/feature171/feature171_1','wb') as fp:
    pickle.dump(f1,fp)
with open('/cshome/kzhou3/Data/feature/feature171/feature171_2','wb') as fp:
    pickle.dump(f2,fp)
with open('/cshome/kzhou3/Data/feature/feature171/feature171_3','wb') as fp:
    pickle.dump(f3,fp)
with open('/cshome/kzhou3/Data/feature/feature171/feature171_4','wb') as fp:
    pickle.dump(f4,fp)
print "=====Total time: " + str((end - start)/60) + " minutes ======"
