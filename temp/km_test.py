import numpy as np
import cPickle as pickle
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

path = '/cshome/kzhou3/Data/feature/feature3/'
files = sorted(os.listdir(path))
with open(path + '1','rb') as fp:
	X = pickle.load(fp)
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=5, min_samples=10).fit(X)
labels = db.labels_
print labels
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
