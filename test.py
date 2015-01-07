f1= open('./submission/19Feat_100class.csv', 'r')
f2 = open('./submission/19Feat_100class.csv', 'r')
f3 = open('./submission/19Feat_100classBinary.csv', 'w')
f1.readline()
f2.readline()
f3.write('driver_trip,prob\n')
# 2737
for i in range(1, 2737):
    array = []
    flag = 0.0
    for j in range(1, 201):
        array.append(float(f1.readline().split(',')[1]))
    array = sorted(array)
    flag = array[10]
    for j in range(1, 201):
        line = f2.readline().split(',')
        if float(line[1]) < flag:
            f3.write(line[0] + ',0\n')
        else:
            f3.write(line[0] + ',1\n')
f1.close()
f2.close()
f3.close()
