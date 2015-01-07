import os
import time
import cPickle as pickle
from getFeature import *

directory = '/cshome/kzhou3/Data/drivers/'
files = sorted(os.listdir(directory))

start = time.time()
c = 0.0
for f in files[2048:]:
    directory = '/cshome/kzhou3/Data/drivers/' + f + '/'
    feature = getFeature(directory)
    # save to pickle
    with open('/cshome/kzhou3/Data/feature/' + f,'wb') as fp:
        pickle.dump(feature,fp)
    c += 1.0
    print "===== " + f + " DONE ====== " + str(c / 2736.0)

end = time.time()
print "=====Total time: " + str(end - start) + " ======"
