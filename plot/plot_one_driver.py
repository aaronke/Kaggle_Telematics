import os
import matplotlib.pyplot as plt

directory = '/cshome/kzhou3/Data/drivers/3/'
files = sorted(os.listdir(directory))
for one_file in files:
    f = open(directory + one_file)
    count = 0
    x = []
    y = []
    f.readline() # skip the first line
    for line in f:
        xy = line.split(',')
        x.append(float(xy[0]))
        y.append(float(xy[1]))
        count += 1
    print count - 1 # this is the total trip time in seconds
    plt.plot(x,y)
    f.close()
plt.show()

