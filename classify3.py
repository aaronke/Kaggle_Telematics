import cPickle as pickle
import os
import time
from sklearn.ensemble import GradientBoostingClassifier as GBRT
import numpy as np
from sklearn.covariance import EmpiricalCovariance

directory = '/cshome/kzhou3/Data/feature/feature3/'
directory2 = '/cshome/kzhou3/Data/feature/feature3/'
files = sorted(os.listdir(directory))
output = open('/cshome/kzhou3/Dropbox/Telematics/submission/GBRT_Out_only_Feat3.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = len(files)
for f in files:
    # load the X to predict nor 3
    with open(directory2 + f,'rb') as fp:
        X_predict = pickle.load(fp)
    # Round 1: get top 100 relavent trips
    X = []
    with open(directory + f,'rb') as fp:
        X += pickle.load(fp)
    X = np.array(X)
    emp_cov = EmpiricalCovariance().fit(X)
    mah = emp_cov.mahalanobis(X)
    MIN = mah.min()
    MAX = mah.max()
    scores = np.array([(MAX - i)/(MAX - MIN) for i in mah])
    scoresII = np.sort(scores)
    # only use the top 140 as training data	
    flag = scoresII[60]
    X_new = []
    for i in range(0,200):
        if scores[i] >= flag:
            X_new.append(X_predict[i])
    X = X_new[:140]
    y = [0]*140 + [1]*800
    # Round 2: train the rest
    for i in range(1,101):
        with open(directory2 + files[(c + i)%size],'rb') as fp:
            X += pickle.load(fp)[:8]
#    clf = RandomForest(n_estimators=250, max_features=4, max_depth=None, min_samples_split=1)
    clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3, random_state=0, subsample = 0.6)
    clf.fit(X, y)
    scores = clf.predict_proba(X_predict)[:,0]
    for i in range(1,201):
        output.write(f + '_' + str(i) + ',' + str(scores[i - 1]) + '\n')
    c += 1
    if c%20 == 0:
        temp = time.time()
        time_remain = int((temp - start)/c*(2736 - c)/60)
        print "------------Remaining time: " + str(time_remain) + " minutes. ------"
    print "===== " + f + " DONE ====== " + str(c / 2736.0*100) + "% completed"

end = time.time()
output.close()
print "=====Total time: " + str((end - start)/60) + " minutes ======"
