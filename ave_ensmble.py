import numpy as np

f1 = open('./ensemble/ensemble52.csv', 'r')
f2 = open('./ensemble/RF_61_2V2.csv', 'r')
f3 = open('./ensemble/ensemble5.csv', 'r')
f4 = open('./ensemble/ensemble52_calib.csv', 'r')
f5 = open('./ensemble/ke.csv', 'r')
f6 = open('./ensemble/ensemble53.csv', 'w')
f1.readline()
f2.readline()
f3.readline()
f4.readline()
f5.readline()
f6.write('driver_trip,prob\n')

for line in f1:
    line = line.split(',')
    a=float(line[1])*1.2
    b=float(f2.readline().split(',')[1])
    c=float(f3.readline().split(',')[1])*0.9
    d=float(f4.readline().split(',')[1])
    e=float(f5.readline().split(',')[1])
    f6.write(line[0] + ',' + str((a+b+c+d+e)/5.0) + '\n')
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
