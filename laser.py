import random

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
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)

    plt.show()



def printDifObstacle(name1,name2):
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\file\\点云\\" + name1 + ".txt", encoding='UTF-8')
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

    ax.scatter(x, y, z,c='blue')
    file_obj = open("E:\\file\\点云\\" + name2 + ".txt", encoding='UTF-8')
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

    ax.scatter(x, y, z, c='red')
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)

    plt.show()
def printObstacles(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    for item in arr:
        c = randomcolor()
        file_obj = open("E:\\file\\点云\\" + item + ".txt", encoding='UTF-8')
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

        ax.scatter(x, y, z,c)


    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    print(plt)
    plt.show()

def randomcolor():
     colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
     colorArr=['000000','0000FF','00008B','#808080','#FF0000']
     # color = ""
     i=random.randint(0, len(colorArr)-1)
     print(i)
     color=colorArr[i]
     del colorArr[i]
    # for i in range(6):
    #     color += colorArr[random.randint(0,14)]
     return "#"+color
if __name__ == '__main__':
    # printDifObstacle('2020-12-31-10-15-18点云分离ground','2020-12-31-10-21-56障碍物')
    printObstacle("2020-12-31-15-31-50障碍物")
    arr=["2020-12-31-10-15-18点云分离obbA","2020-12-31-10-15-18点云分离obbB","2020-12-31-10-15-18点云分离obbD","2020-12-31-10-15-18点云分离obbE","2020-12-31-10-15-18点云分离球体"]

    # printObstacles(arr)
