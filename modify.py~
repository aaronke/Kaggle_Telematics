import numpy as np

f1 = open('./submission/19Feat_100class.csv', 'r')
f2 = open('./submission/19Feat_100class.csv', 'r')
f3 = open('./submission/R_19Feat_100class_M.csv', 'w')
f1.readline()
f2.readline()
f3.write('driver_trip,prob\n')
# 2737
for i in range(1, 2737):
    array = []
    for j in range(1, 201):
        array.append(float(f1.readline().split(',')[1]))
    array = np.array(array)
    MIN = array.min()
    MAX = array.max()
    for j in range(1, 201):
        line = f2.readline().split(',')
        predict = (MAX - float(line[1]))/(MAX - MIN)
        f3.write(line[0] + ',' + str(predict) + '\n')
f1.close()
f2.close()
f3.close()
