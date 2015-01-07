import cPickle as pickle
import os
import time
from sklearn.svm import NuSVC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest
import numpy as np

directory = '/cshome/kzhou3/Data/feature/feature3nor/'
files = sorted(os.listdir(directory))
output = open('/cshome/kzhou3/Dropbox/Telematics/submission/GBRT_para_half_Feat3_nor.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = len(files)
for f in files:
    # load the X to predict
    with open(directory + f,'rb') as fp:
        X_predict = pickle.load(fp)
    X = []
    y = [0]*200 + [1]*2000
    with open(directory + f,'rb') as fp:
        X += pickle.load(fp)
    for i in range(1,201):
        with open(directory + files[(c + i)%size],'rb') as fp:
            X += pickle.load(fp)[:10]
#    clf = RandomForest(n_estimators=250, max_features=4, max_depth=None, min_samples_split=1)
    clf = GBRT(n_estimators=250, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=3, random_state=0, subsample = 0.6)
    clf.fit(X, y)
    scores = clf.predict_proba(X[:200])[:,0]
    # only use the top 100 as training data
    scoresII = np.sort(scores)
    flag = scoresII[100]
    X_new = []
    for i in range(0,200):
        if scores[i] >= flag:
            X_new.append(X[i])
    X = X_new[:100] + X[200:]
    y = y[:100] + y[200:]
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
