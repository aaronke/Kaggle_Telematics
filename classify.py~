import cPickle as pickle
import os
import time
from sklearn.svm import NuSVC
from sklearn.ensemble import GradientBoostingClassifier as GBRT
from sklearn.ensemble import RandomForestClassifier as RandomForest

directory = '/cshome/kzhou3/Data/feature/'
files = sorted(os.listdir(directory))
output = open('/cshome/kzhou3/Dropbox/Telematics/submission/RF_3.csv', 'w')
output.write('driver_trip,prob\n')

start = time.time()
c = 0
size = len(files)
for f in files:
    X = []
    y = [0]*200 + [1]*500
    with open(directory + f,'rb') as fp:
        X += pickle.load(fp)
    for i in range(1,11):
        with open(directory + files[(c + i)%size],'rb') as fp:
            X += pickle.load(fp)[:50]
    clf = RandomForest(n_estimators=10).fit(X, y)
    #clf = GBRT(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
    scores = clf.predict_proba(X[:200])
    for i in range(1,201):
        output.write(f + '_' + str(i) + ',' + str(scores[i - 1][0]) + '\n')
    c += 1
    print "===== " + f + " DONE ====== " + str(c / 2736.0)

end = time.time()
output.close()
print "=====Total time: " + str(end - start) + " ======"
