import cPickle as pickle

with open('/cshome/kzhou3/Data/feature/feature2/1','rb') as fp:
	feature = pickle.load(fp)

for f in feature:
    print f
