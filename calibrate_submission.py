import numpy as np
import operator
"""
from scipy.io import loadmat

path = '/cshome/kzhou3/Data/feature/featureK/'
name = loadmat(path + 'name.mat')
name = name['Sort_Names'][0]
"""
f1 = open('./submission/RF_61_2V2.csv', 'r')
f2 = open('./submission/RF_61_2V2_calib.csv', 'w')
f1.readline()
f2.write('driver_trip,prob\n')
# 2737
for i in range(1, 2737):
    dict = {}
    name = ''
    for j in range(1, 201):
        line = f1.readline().split(',')
        name = line[0]
        dict[j] = float(line[1])
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
    k = 0
    for trip_id in sorted_dict:
        f2.write(name.split('_')[0] + '_' + str(trip_id[0]) + ',' + str(0.005*k) + '\n')
        k += 1
f1.close()
f2.close()
