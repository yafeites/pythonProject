import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as  np
from scipy import interpolate
import math
def printObstacle(name):
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\file\\点云\\" + name + ".txt", encoding='UTF-8')
    x = []
    y = []
    z = []
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        print(arr)
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))

    ax.scatter(x,y,z)

    plt.show()




if __name__ == '__main__':
    printObstacle("2020-12-26-10-29-20obstacle1")
