from scipy.io import loadmat
import numpy as np

a = loadmat('/cshome/kzhou3/Data/feature/featureK/driver_features_42.mat')
#a = loadmat('/cshome/kzhou3/Data/feature/featureK/Sort_Names.mat')

b = a['drivers_features']
#b = a['Sort_Names']
#print type(b)
print b.shape
print b[547199:547205]
