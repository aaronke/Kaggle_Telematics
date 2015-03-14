import numpy as np

f1 = open('./ensemble/7_to_1/90464.csv', 'r') # 171 feature, 2v2_RF, 1888 estimator, auto max feature
f2 = open('./ensemble/7_to_1/89400.csv', 'r') # 75 feature, 2v25_RF, 550 estimator, 8 max feature
f3 = open('./ensemble/7_to_1/88035.csv', 'r') # lcl's
f4 = open('./ensemble/7_to_1/85427.csv', 'r') # 4V4_RF, 550 estimator
f5 = open('./ensemble/7_to_1/84425.csv', 'r') # my 19 feature, 1v100 class
f6 = open('./ensemble/7_to_1/80534.csv', 'r') # 171 feature, pure LR testing
f7 = open('./ensemble/7_to_1/87188_cal.csv', 'r') # GBRT, 171, 550 est, calibrated

f8 = open('./ensemble/7_to_1/the_1_from_4.csv', 'w')

f1.readline()
f2.readline()
f3.readline()
f4.readline()
f5.readline()
f6.readline()
f7.readline()
f8.write('driver_trip,prob\n')

for line in f1:
    line=line.split(',')
    a=float(line[1])
    b=float(f2.readline().split(',')[1])
    c=float(f3.readline().split(',')[1])
    d=float(f4.readline().split(',')[1])
    e=float(f5.readline().split(',')[1])
    f=float(f6.readline().split(',')[1])
    g=float(f7.readline().split(',')[1])
    f8.write(line[0] + ',' + str((a+b+c+g)/4.0) + '\n')

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()
f8.close()
