import os
import time
import numpy as np
import cPickle as pickle

path = '/cshome/kzhou3/Data/feature/feature61/'
with open(path + 'feature61','rb') as fp:
    feature = pickle.load(fp)

f = open('./submission/RF_61_2V2_calib.csv', 'r')
f.readline()

feature_half = np.array([[0]*61])
for k in range(0, 2736):
    for j in range(0,100):
        f.readline()
    X_new = []
    for j in range(0,100):
        line = f.readline().split(',') # 'driverID_tripID, rank'
        name = line[0].split('_') # 'driverID_tripID'
        X_new.append(feature[k*200 + int(name[1]) - 1])
    feature_half = np.concatenate((feature_half, np.array(X_new)))
    if k%20 == 0:
        print "DONE" + str(k)

feature_half = feature_half[1:]
print feature_half.shape
with open('/cshome/kzhou3/Data/feature/feature61/feature61_half','wb') as fp:
    pickle.dump(feature_half,fp)
f.close()
