import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets
from mpl_toolkits.mplot3d.axes3d import Axes3D

path = '/cshome/kzhou3/Data/feature/feature3/'
with open(path + '14','rb') as fp:
	feature1 = pickle.load(fp)
with open(path + '103','rb') as fp:
	feature2 = pickle.load(fp)
with open(path + '161','rb') as fp:
	feature3 = pickle.load(fp)

X = feature1 + feature2 + feature3
y = [0]*200 + [1]*200 + [2]*200
labels = ['driver1', 'driver2', 'driver3']
X = np.array(X)
y = np.array(y)
labels = np.array(labels)
"""
iris = datasets.load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
"""
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

x_surf = [X[:, 0].min(), X[:, 0].max(),
          X[:, 0].min(), X[:, 0].max()]
y_surf = [X[:, 0].max(), X[:, 0].max(),
          X[:, 0].min(), X[:, 0].min()]
x_surf = np.array(x_surf)
y_surf = np.array(y_surf)
plt.show()
