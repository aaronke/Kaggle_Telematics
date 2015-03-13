# change lcl's feature into pickle list format, 200*96
import os
import time
import cPickle as pickle

directory = '/cshome/kzhou3/Data/feature/feature_lcl/features/'
files = sorted(os.listdir(directory))

start = time.time()
c = 0.0
for f in files:
    this_file = '/cshome/kzhou3/Data/feature/feature_lcl/features/' + f
    driver = open(this_file)
    driver.readline() # skip first line
    feature = []
    for line in driver:
        data = [float(i) for i in line.strip().split(',')]
        feature.append(data)
    # save to pickle
    f = f.strip().split('.')[0] # fetch the index
    with open('/cshome/kzhou3/Data/feature/feature96_lcl/' + f,'wb') as fp:
        pickle.dump(feature,fp)
    c += 1.0
    print "===== " + f + " DONE ====== " + str(c / 2736.0)

end = time.time()
print "=====Total time: " + str(end - start) + " ======"
