import numpy as np

f1 = open('./submission/final/305_ada_RF_2222_52.csv', 'r')
f2 = open('./submission/final/sepa_feat_1.csv', 'r')
f3 = open('./submission/final/sepa_feat_2.csv', 'r')
f4 = open('./submission/final/sepa_feat_3.csv', 'r')
f5 = open('./ensemble/RF_2v2_sorted.csv', 'r')
f6 = open('all_sepa_together.csv', 'w')
f1.readline()
f2.readline()
f3.readline()
f4.readline()
f5.readline()
f6.write('driver_trip,prob\n')

for line in f1:
    line = line.split(',')
    a=float(line[1])
    b=float(f2.readline().split(',')[1])
    c=float(f3.readline().split(',')[1])
    d=float(f4.readline().split(',')[1])
    e=float(f5.readline().split(',')[1])
    f6.write(line[0] + ',' + str((a+b+c+d)/4.0) + '\n')
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
