f1 = open('./submission/3.csv', 'r')
f2 = open('./submission/R_3.csv', 'w')
f1.readline()
f2.write('driver_trip,prob\n')

for line in f1:
    line = line.split(',')
    predict = (1-float(line[1]))
    f2.write(line[0] + ',' + str(predict) + '\n')
f1.close()
f2.close()
