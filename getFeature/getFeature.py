import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS
import numpy as np

# get speed
def getSpeed(p1, p2):
    speed = p2 - p1
    return np.sqrt((speed*speed).sum())

# get angle in range(0,180)
def getAngle(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    v1_mod = np.sqrt((v1*v1).sum()) + 0.001
    v2_mod = np.sqrt((v2*v2).sum()) + 0.001
    cos_angle = dot / v1_mod / v2_mod
    angle = np.arccos(cos_angle)
    return 180 * angle / 3.1415

# directory = '/cshome/kzhou3/Data/drivers/1/'
def getFeature(directory):
    feature = []
    for i in range(1,201):
        trip = []
        seconds = 0
        speed = []
        acce = []
        rotate = []
        
        f = open(directory + str(i) + '.csv')
        
        f.readline() # skip the first line (x, y)
        f.readline() # skip the second line (0, 0)
        start_point = np.array([0,0])
        trip.append(start_point)
        
        line = f.readline() # handle the first move
        x = line.split(',')
        x = np.array([float(x[0]), float(x[1])])
        trip.append(x)
        seconds += 1
        speed.append(np.sqrt((x*x).sum()))
        acce.append(np.sqrt((x*x).sum()))
        rotate.append(0)
    
        time_serial = []
        time_serial.append(seconds)
    
        for line in f:
            x = line.split(',')
            x = np.array([float(x[0]), float(x[1])])
    
            # trip points
            trip.append(x)
    
            # time
            seconds += 1
            time_serial.append(seconds)
    
            # speed
            v = 3.6*getSpeed(trip[-2], trip[-1])
            speed.append(v)
    
            # acce
            acce.append(abs(speed[-2] - speed[-1]))
    
            # rotate
            rotate.append(getAngle(trip[-3], trip[-2], trip[-1]))
        speed = np.array(speed)
        acce = np.array(acce)
        rotate = np.array(rotate)
        one_feature = [seconds, np.mean(speed), np.std(speed), np.mean(acce), np.std(acce), np.mean(rotate), np.std(rotate)]
        feature.append(one_feature)
        f.close()
    return feature

# mean, max, std of speed acce angle
def getFeature2(directory):
    feature = []
    for i in range(1,201):
        trip = []
        seconds = 0
        speed = []
        acce = []
        rotate = []
        
        f = open(directory + str(i) + '.csv')
        
        f.readline() # skip the first line (x, y)
        f.readline() # skip the second line (0, 0)
        start_point = np.array([0,0])
        trip.append(start_point)
        
        line = f.readline() # handle the first move
        x = line.split(',')
        x = np.array([float(x[0]), float(x[1])])
        trip.append(x)
        seconds += 1
        speed.append(np.sqrt((x*x).sum()))
        acce.append(np.sqrt((x*x).sum()))
        rotate.append(0)
    
        time_serial = []
        time_serial.append(seconds)
    
        for line in f:
            x = line.split(',')
            x = np.array([float(x[0]), float(x[1])])
    
            # trip points
            trip.append(x)
    
            # time
            seconds += 1
            time_serial.append(seconds)
    
            # speed
            v = 3.6*getSpeed(trip[-2], trip[-1])
            if v > 150:
                continue;
            speed.append(v)
    
            # acce
            ac = abs(speed[-2] - speed[-1])
            if ac > 15:
                 continue;
            acce.append(ac)

            # rotate
            rotate.append(getAngle(trip[-3], trip[-2], trip[-1]))
        speed = np.array(speed)
        acce = np.array(acce)
        rotate = np.array(rotate)
        one_feature = [seconds, np.mean(speed), np.std(speed), np.amax(speed), np.mean(acce), np.std(acce), np.amax(acce), np.mean(rotate), np.std(rotate), np.amax(rotate)]
        feature.append(one_feature)
        f.close()
    return feature

# mean, max, std of speed acce angle
def getFeature2nor(directory):
    feature = np.empty([200,10])
    for i in range(1,201):
        trip = []
        seconds = 0
        speed = []
        acce = []
        rotate = []
        
        f = open(directory + str(i) + '.csv')
        
        f.readline() # skip the first line (x, y)
        f.readline() # skip the second line (0, 0)
        start_point = np.array([0,0])
        trip.append(start_point)
        
        line = f.readline() # handle the first move
        x = line.split(',')
        x = np.array([float(x[0]), float(x[1])])
        trip.append(x)
        seconds += 1
        speed.append(np.sqrt((x*x).sum()))
        acce.append(np.sqrt((x*x).sum()))
        rotate.append(0)
    
        time_serial = []
        time_serial.append(seconds)
    
        for line in f:
            x = line.split(',')
            x = np.array([float(x[0]), float(x[1])])
    
            # trip points
            trip.append(x)
    
            # time
            seconds += 1
            time_serial.append(seconds)
    
            # speed
            v = 3.6*getSpeed(trip[-2], trip[-1])
            if v > 150:
                continue;
            speed.append(v)
    
            # acce
            ac = abs(speed[-2] - speed[-1])
            if ac > 15:
                 continue;
            acce.append(ac)

            # rotate
            rotate.append(getAngle(trip[-3], trip[-2], trip[-1]))
        speed = np.array(speed)
        acce = np.array(acce)
        rotate = np.array(rotate)
        feature[i - 1] = [float(seconds), np.mean(speed), np.std(speed), np.amax(speed), np.mean(acce), np.std(acce), np.amax(acce), np.mean(rotate), np.std(rotate), np.amax(rotate)]
        f.close()
    scaler = SS().fit(feature)
    feature = scaler.transform(feature)
    return np.ndarray.tolist(feature)

# mean std max of speed_high & speed_low, median of speed and it's occupation time, nomarlaztion
def getFeature3(directory):
    feature = []
    for i in range(1,201):
        trip = []
        seconds = 0
        speed = []
        acce = []
        rotate = []
        speed2 = []
        acce2 = []
        rotate2 = []
        speedI = []
        
        f = open(directory + str(i) + '.csv')
        
        f.readline() # skip the first line (x, y)
        f.readline() # skip the second line (0, 0)
        start_point = np.array([0,0])
        trip.append(start_point)
        
        line = f.readline() # handle the first move
        x = line.split(',')
        x = np.array([float(x[0]), float(x[1])])
        trip.append(x)
        seconds += 1
        v = np.sqrt((x*x).sum())
        speed.append(v)
        speedI.append(v)
        if (v < 8):
            acce.append(v)
        else:
            acce2.append(v)
        rotate.append(0)
    
        for line in f:
            x = line.split(',')
            x = np.array([float(x[0]), float(x[1])])
    
            # trip points
            trip.append(x)
    
            # time
            seconds += 1
    
            # speed
            v = 3.6*getSpeed(trip[-2], trip[-1])
            speedI.append(v)
            if v > 50:
                speed.append(v)
            else:
                speed2.append(v)
    
            # acce
            ac = abs(speedI[-2] - speedI[-1])
            if ac > 8:
                acce.append(ac)
            else:
                acce2.append(ac)

            # rotate
            angle = getAngle(trip[-3], trip[-2], trip[-1])
            if angle > 44:
                rotate.append(angle)
            else:
                rotate2.append(angle)
        speedI = [int(i) for i in speedI]
        speedI = np.array(speedI)
        s_median = np.median(speedI)
        s_time = 0
        for s in speedI:
            if s < s_median + 8 and s > s_median - 8:
                s_time += 1
        s_time = float(s_time)/seconds*100
        
        # deal with empty lists
        if len(speed) == 0:
            speed = [0]
        if len(speed2) == 0:
            speed2 = [0]
        if len(acce) == 0:
            acce = [0]
        if len(acce2) == 0:
            acce2 = [0]
        if len(rotate) == 0:
            rotate = [0]
        if len(rotate2) == 0:
            rotate2 = [0]
        speed = np.array(speed)
        acce = np.array(acce)
        rotate = np.array(rotate)
        speed2 = np.array(speed2)
        acce2 = np.array(acce2)
        rotate2 = np.array(rotate2)
        one_feature = [float(seconds), np.mean(speed), np.std(speed), np.amax(speed), np.mean(acce), np.std(acce), np.amax(acce), np.mean(rotate), np.std(rotate), np.mean(speed2), np.std(speed2), np.amax(speed2), np.mean(acce2), np.std(acce2), np.amax(acce2), np.mean(rotate2), np.std(rotate2), s_median, s_time]
        feature.append(one_feature)
        f.close()
    return feature

# mean std max of speed_high & speed_low, median of speed and it's occupation time, nomarlaztion
def getFeature3nor(directory):
    feature = np.empty([200,19])
    for i in range(1,201):
        trip = []
        seconds = 0
        speed = []
        acce = []
        rotate = []
        speed2 = []
        acce2 = []
        rotate2 = []
        speedI = []
        
        f = open(directory + str(i) + '.csv')
        
        f.readline() # skip the first line (x, y)
        f.readline() # skip the second line (0, 0)
        start_point = np.array([0,0])
        trip.append(start_point)
        
        line = f.readline() # handle the first move
        x = line.split(',')
        x = np.array([float(x[0]), float(x[1])])
        trip.append(x)
        seconds += 1
        v = np.sqrt((x*x).sum())
        speed.append(v)
        speedI.append(v)
        if (v < 8):
            acce.append(v)
        else:
            acce2.append(v)
        rotate.append(0)
    
        for line in f:
            x = line.split(',')
            x = np.array([float(x[0]), float(x[1])])
    
            # trip points
            trip.append(x)
    
            # time
            seconds += 1
    
            # speed
            v = 3.6*getSpeed(trip[-2], trip[-1])
            speedI.append(v)
            if v > 50:
                speed.append(v)
            else:
                speed2.append(v)
    
            # acce
            ac = abs(speedI[-2] - speedI[-1])
            if ac > 8:
                acce.append(ac)
            else:
                acce2.append(ac)

            # rotate
            angle = getAngle(trip[-3], trip[-2], trip[-1])
            if angle > 44:
                rotate.append(angle)
            else:
                rotate2.append(angle)
        speedI = [int(s) for s in speedI]
        speedI = np.array(speedI)
        s_median = np.median(speedI)
        s_time = 0
        for s in speedI:
            if s < s_median + 8 and s > s_median - 8:
                s_time += 1
        s_time = float(s_time)/seconds*100

        # deal with empty lists
        if len(speed) == 0:
            speed = [0]
        if len(speed2) == 0:
            speed2 = [0]
        if len(acce) == 0:
            acce = [0]
        if len(acce2) == 0:
            acce2 = [0]
        if len(rotate) == 0:
            rotate = [0]
        if len(rotate2) == 0:
            rotate2 = [0]
        speed = np.array(speed)
        acce = np.array(acce)
        rotate = np.array(rotate)
        speed2 = np.array(speed2)
        acce2 = np.array(acce2)
        rotate2 = np.array(rotate2)
        feature[i - 1] = [float(seconds), np.mean(speed), np.std(speed), np.amax(speed), np.mean(acce), np.std(acce), np.amax(acce), np.mean(rotate), np.std(rotate), np.mean(speed2), np.std(speed2), np.amax(speed2), np.mean(acce2), np.std(acce2), np.amax(acce2), np.mean(rotate2), np.std(rotate2), s_median, s_time]
        f.close()
    scaler = SS().fit(feature)
    feature = scaler.transform(feature)
    return np.ndarray.tolist(feature)

if __name__ == "__main__":
    directory = '/cshome/kzhou3/Data/drivers/1/'
    print getFeature3nor(directory)
"""
    feature = []
    for i in range(14,15):
        trip = []
        seconds = 0
        speed = []
        acce = []
        rotate = []
        speed2 = []
        acce2 = []
        rotate2 = []
        speedI = []
        
        f = open(directory + str(i) + '.csv')
        
        f.readline() # skip the first line (x, y)
        f.readline() # skip the second line (0, 0)
        start_point = np.array([0,0])
        trip.append(start_point)
        
        line = f.readline() # handle the first move
        x = line.split(',')
        x = np.array([float(x[0]), float(x[1])])
        trip.append(x)
        seconds += 1
        v = np.sqrt((x*x).sum())
        speed.append(v)
        speedI.append(v)
        if (v < 8):
            acce.append(v)
        else:
            acce2.append(v)
        rotate.append(0)

    
        for line in f:
            x = line.split(',')
            x = np.array([float(x[0]), float(x[1])])

            # trip points
            trip.append(x)
    
            # time
            seconds += 1
    
            # speed
            v = 3.6*getSpeed(trip[-2], trip[-1])
            speedI.append(v)
            if v > 50:
                speed.append(v)
            else:
                speed2.append(v)
    
            # acce
            ac = abs(speedI[-2] - speedI[-1])
            if ac > 8:
                 acce.append(ac)
            else:
                acce2.append(ac)

            # rotate
            angle = getAngle(trip[-3], trip[-2], trip[-1])
            if angle > 44:
                 rotate.append(angle)
            else:
                rotate2.append(angle)
        speedI = [int(i) for i in speedI]
        aa, = plt.plot(speedI)
        bb, = plt.plot(acce2)
        cc, = plt.plot(rotate2)
        plt.legend([aa, bb, cc], ['speed', 'acce', 'rotate'])
        plt.show()
"""
