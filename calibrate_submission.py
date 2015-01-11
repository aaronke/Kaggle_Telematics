import numpy as np
import operator
from scipy.io import loadmat

path = '/cshome/kzhou3/Data/feature/featureK/'
name = loadmat(path + 'name.mat')
name = name['Sort_Names'][0]

f1 = open('./submission/GBRT_Ke_42_2V8.csv', 'r')
f2 = open('./submission/GBRT_Ke_42_2V8_calib.csv', 'w')
f1.readline()
f2.write('driver_trip,prob\n')
# 2737
for i in range(1, 2737):
    dict = {}
    for j in range(1, 201):
        dict[j] = float(f1.readline().split(',')[1])
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
    k = 0
    for trip_id in sorted_dict:
        f2.write(str(name[i - 1]) + '_' + str(trip_id[0]) + ',' + str(0.005*k) + '\n')
        k += 1
f1.close()
f2.close()