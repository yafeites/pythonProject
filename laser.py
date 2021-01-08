import random

import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D
import numpy as  np
from scipy import interpolate
import math


def printOct(halfLength1, point, vector, ax):
    halfLength=[halfLength1[0]+15,halfLength1[1]+15,halfLength1[2]+15]
    print(vector)
    print(point)
    pointA = [[point[0], point[1], point[2]] for i in range(8)]
    print(vector[0][0])
    pointA[0][0] += halfLength[0] * vector[0][0]
    pointA[0][1] += halfLength[0] * vector[0][1]
    pointA[0][2] += halfLength[0] * vector[0][2]
    pointA[0][0] += halfLength[1] * vector[1][0]
    pointA[0][1] += halfLength[1] * vector[1][1]
    pointA[0][2] += halfLength[1] * vector[1][2]
    pointA[0][0] += halfLength[2] * vector[2][0]
    pointA[0][1] += halfLength[2] * vector[2][1]
    pointA[0][2] += halfLength[2] * vector[2][2]
    # print(pointA[0][0])
    pointA[1][0] += - halfLength[0] * vector[0][0]
    pointA[1][1] += - halfLength[0] * vector[0][1]
    pointA[1][2] += - halfLength[0] * vector[0][2]
    pointA[1][0] += + halfLength[1] * vector[1][0]
    pointA[1][1] += + halfLength[1] * vector[1][1]
    pointA[1][2] += + halfLength[1] * vector[1][2]
    pointA[1][0] += + halfLength[2] * vector[2][0]
    pointA[1][1] += + halfLength[2] * vector[2][1]
    pointA[1][2] += + halfLength[2] * vector[2][2]

    pointA[2][0] += + halfLength[0] * vector[0][0]
    pointA[2][1] += + halfLength[0] * vector[0][1]
    pointA[2][2] += + halfLength[0] * vector[0][2]
    pointA[2][0] += - halfLength[1] * vector[1][0]
    pointA[2][1] += - halfLength[1] * vector[1][1]
    pointA[2][2] += - halfLength[1] * vector[1][2]
    pointA[2][0] += + halfLength[2] * vector[2][0]
    pointA[2][1] += + halfLength[2] * vector[2][1]
    pointA[2][2] += + halfLength[2] * vector[2][2]

    pointA[3][0] += + halfLength[0] * vector[0][0]
    pointA[3][1] += + halfLength[0] * vector[0][1]
    pointA[3][2] += + halfLength[0] * vector[0][2]
    pointA[3][0] += + halfLength[1] * vector[1][0]
    pointA[3][1] += + halfLength[1] * vector[1][1]
    pointA[3][2] += + halfLength[1] * vector[1][2]
    pointA[3][0] += - halfLength[2] * vector[2][0]
    pointA[3][1] += - halfLength[2] * vector[2][1]
    pointA[3][2] += - halfLength[2] * vector[2][2]

    pointA[4][0] += - halfLength[0] * vector[0][0]
    pointA[4][1] += - halfLength[0] * vector[0][1]
    pointA[4][2] += - halfLength[0] * vector[0][2]
    pointA[4][0] += - halfLength[1] * vector[1][0]
    pointA[4][1] += - halfLength[1] * vector[1][1]
    pointA[4][2] += - halfLength[1] * vector[1][2]
    pointA[4][0] += + halfLength[2] * vector[2][0]
    pointA[4][1] += + halfLength[2] * vector[2][1]
    pointA[4][2] += + halfLength[2] * vector[2][2]

    pointA[5][0] += - halfLength[0] * vector[0][0]
    pointA[5][1] += - halfLength[0] * vector[0][1]
    pointA[5][2] += - halfLength[0] * vector[0][2]
    pointA[5][0] += + halfLength[1] * vector[1][0]
    pointA[5][1] += + halfLength[1] * vector[1][1]
    pointA[5][2] += + halfLength[1] * vector[1][2]
    pointA[5][0] += - halfLength[2] * vector[2][0]
    pointA[5][1] += - halfLength[2] * vector[2][1]
    pointA[5][2] += - halfLength[2] * vector[2][2]

    pointA[6][0] += + halfLength[0] * vector[0][0]
    pointA[6][1] += + halfLength[0] * vector[0][1]
    pointA[6][2] += + halfLength[0] * vector[0][2]
    pointA[6][0] += - halfLength[1] * vector[1][0]
    pointA[6][1] += - halfLength[1] * vector[1][1]
    pointA[6][2] += - halfLength[1] * vector[1][2]
    pointA[6][0] += - halfLength[2] * vector[2][0]
    pointA[6][1] += - halfLength[2] * vector[2][1]
    pointA[6][2] += - halfLength[2] * vector[2][2]

    pointA[7][0] += - halfLength[0] * vector[0][0]
    pointA[7][1] += - halfLength[0] * vector[0][1]
    pointA[7][2] += - halfLength[0] * vector[0][2]
    pointA[7][0] += - halfLength[1] * vector[1][0]
    pointA[7][1] += - halfLength[1] * vector[1][1]
    pointA[7][2] += - halfLength[1] * vector[1][2]
    pointA[7][0] += - halfLength[2] * vector[2][0]
    pointA[7][1] += - halfLength[2] * vector[2][1]
    pointA[7][2] += - halfLength[2] * vector[2][2]
    print(pointA)

    printLine(pointA,0,1,ax)
    printLine(pointA,4,1,ax)
    printLine(pointA,4,2,ax)
    printLine(pointA,0,2,ax)
    printLine(pointA,5,1,ax)
    printLine(pointA,4,7,ax)
    printLine(pointA,2,6,ax)
    printLine(pointA,0,3,ax)
    printLine(pointA,5,7,ax)
    printLine(pointA,5,3,ax)
    printLine(pointA,7,6,ax)
    printLine(pointA,6,3,ax)

def printLine(pointA,i,j,ax):
            x = np.linspace(float(pointA[i][0]), float(pointA[j][0]), 2)
            y = np.linspace(float(pointA[i][1]), float(pointA[j][1]), 2)
            z = np.linspace(float(pointA[i][2]), float(pointA[j][2]), 2)
            ax.plot(x, y, z, 'red')
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

    ax.scatter(x, y, z, s=1)

    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)

    plt.show()


def printObstacleObb(name):
    arr1 = [[1, 2, 3]]
    arr2 = [[4, 0, 3]]
    # print(np.dot(np.transpose(arr1) ,arr2))
    print(np.dot(arr1 ,np.transpose(arr2)))
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\file\\点云\\" + name + ".txt", encoding='UTF-8')

    x = []
    y = []
    z = []
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        # print(arr)
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
    size = len(x)
    # print(size)
    points = numpy.zeros((size, 3))
    print(np.transpose(points))
    # print(points)
    for i in range(size):
        points[i][0] = x[i]
        points[i][1] = y[i]
        points[i][2] = z[i]
        # print(points[i])

    avg = np.array([0,0,0])
    # print(avg)
    # avg[0] = 0
    # avg[1] = 0
    # avg[2] = 0
    for i in range(size):
        avg[0] += x[i]
        avg[1] += y[i]
        avg[2] += z[i]
    avg[0]/=size
    avg[1]/=size
    avg[2]/=size
    arr=numpy.linalg.eig(numpy.cov(np.transpose(points)))
    print(arr)
    # print(numpy.linalg.eig(numpy.cov(points)))
    # for i in range(size):
    #     for j in range(size):
    #         num_list[i][j] =
    maxL=-100000
    maxW=-100000
    maxH=-100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL=max(maxL,np.dot(point,np.transpose(arr[1][0])))
        minL=min(minL,np.dot(point,np.transpose(arr[1][0])))
        maxW = max(maxW,np.dot(point, np.transpose(arr[1][1])))
        minW = min(minW,np.dot(point, np.transpose(arr[1][1])))
        maxH = max(maxH,np.dot(point, np.transpose(arr[1][2])))
        minH = min(minH,np.dot(point, np.transpose(arr[1][2])))
        # print(minL)
    tr=np.transpose(arr[1])
    print(tr)
    pointA=[(maxL+minL)/2,(maxW+minW)/2,(maxH+minH)/2]

    X=np.dot(pointA,np.transpose(tr[0]))
    Y=np.dot(pointA,np.transpose(tr[1]))
    Z=np.dot(pointA,np.transpose(tr[2]))
    pointB=[X,Y,Z]
    print(X,Y,Z)

    halfLengthA=[(maxL-minL)/2,(maxW-minW)/2,(maxH-minH)/2]

    printOct(halfLengthA, pointB, arr[1], ax)

    ax.scatter(x, y, z, s=1)
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    plt.show()

def printObstacleObbByNotZ(name,ax):
    # arr1 = [[1, 2, 3]]
    # arr2 = [[4, 0, 3]]
    # print(np.dot(np.transpose(arr1) ,arr2))
    # print(np.dot(arr1 ,np.transpose(arr2)))


    file_obj = open("E:\\file\\点云\\" + name + ".txt", encoding='UTF-8')

    x = []
    y = []
    z = []
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        # print(arr)
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
    size = len(x)
    # print(size)
    points = numpy.zeros((size, 3))
    pointsNotZ = numpy.zeros((size, 2))

    print(np.transpose(points))
    # print(points)
    for i in range(size):
        points[i][0] = x[i]
        points[i][1] = y[i]
        points[i][2] = z[i]
        # print(points[i])
        pointsNotZ[i][0] = x[i]
        pointsNotZ[i][1] = y[i]
    avg = np.array([0,0])
    # print(avg)
    # avg[0] = 0
    # avg[1] = 0
    # avg[2] = 0
    for i in range(size):
        avg[0] += x[i]
        avg[1] += y[i]
        # avg[2] += z[i]
    avg[0]/=size
    avg[1]/=size
    # avg[2]/=size
    arr=numpy.linalg.eig(numpy.cov(np.transpose(pointsNotZ)))
    print(arr)
    # print(numpy.linalg.eig(numpy.cov(points)))
    maxL=-100000
    maxW=-100000
    maxH=-100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL=max(maxL,np.dot(point[:2],np.transpose(arr[1][0])))
        minL=min(minL,np.dot(point[:2],np.transpose(arr[1][0])))
        maxW = max(maxW,np.dot(point[:2], np.transpose(arr[1][1])))
        minW = min(minW,np.dot(point[:2], np.transpose(arr[1][1])))
        maxH = max(maxH,point[2])
        minH = min(minH,point[2])
        # print(minL)
    tr=np.transpose(arr[1])
    print(tr)
    pointA=[(maxL+minL)/2,(maxW+minW)/2,(maxH+minH)/2]

    X=np.dot(pointA[:2],np.transpose(tr[0]))
    Y=np.dot(pointA[:2],np.transpose(tr[1]))
    Z=pointA[2]
    pointB=[X,Y,Z]
    print(X,Y,Z)
    arrVector=arr[1]
    print(arrVector)
    halfLengthA=[(maxL-minL)/2,(maxW-minW)/2,(maxH-minH)/2]
    vector=[[arrVector[0][0],arrVector[0][1],0],[arrVector[1][0],arrVector[1][1],0],[0,0,1]]
    print(vector)
    printOct(halfLengthA, pointB, vector, ax)

    ax.scatter(x, y, z, s=1)
    # plt.xlim(1000, 2500)
    # plt.ylim(-1000, 1000)
    # plt.show()

def printObstacleObbByNotZTest(name):
    # arr1 = [[1, 2, 3]]
    # arr2 = [[4, 0, 3]]
    # print(np.dot(np.transpose(arr1) ,arr2))
    # print(np.dot(arr1 ,np.transpose(arr2)))

    fig = plt.figure()
    ax = Axes3D(fig)
    file_obj = open("E:\\file\\点云\\" + name + ".txt", encoding='UTF-8')

    x = []
    y = []
    z = []
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        # print(arr)
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
    size = len(x)
    # print(size)
    points = numpy.zeros((size, 3))
    pointsNotZ = numpy.zeros((size, 2))

    print(np.transpose(points))
    # print(points)
    for i in range(size):
        points[i][0] = x[i]
        points[i][1] = y[i]
        points[i][2] = z[i]
        # print(points[i])
        pointsNotZ[i][0] = x[i]
        pointsNotZ[i][1] = y[i]
    avg = np.array([0,0])
    # print(avg)
    # avg[0] = 0
    # avg[1] = 0
    # avg[2] = 0
    for i in range(size):
        avg[0] += x[i]
        avg[1] += y[i]
        # avg[2] += z[i]
    avg[0]/=size
    avg[1]/=size
    # avg[2]/=size
    arr=numpy.linalg.eig(numpy.cov(np.transpose(pointsNotZ)))
    print(arr)
    # print(numpy.linalg.eig(numpy.cov(points)))
    maxL=-100000
    maxW=-100000
    maxH=-100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL=max(maxL,np.dot(point[:2],np.transpose(arr[1][0])))
        minL=min(minL,np.dot(point[:2],np.transpose(arr[1][0])))
        maxW = max(maxW,np.dot(point[:2], np.transpose(arr[1][1])))
        minW = min(minW,np.dot(point[:2], np.transpose(arr[1][1])))
        maxH = max(maxH,point[2])
        minH = min(minH,point[2])
        # print(minL)
    tr=np.transpose(arr[1])
    print(tr)
    pointA=[(maxL+minL)/2,(maxW+minW)/2,(maxH+minH)/2]

    X=np.dot(pointA[:2],np.transpose(tr[0]))
    Y=np.dot(pointA[:2],np.transpose(tr[1]))
    Z=pointA[2]
    pointB=[X,Y,Z]
    print(X,Y,Z)
    arrVector=arr[1]
    print(arrVector)
    halfLengthA=[(maxL-minL)/2,(maxW-minW)/2,(maxH-minH)/2]
    vector=[[arrVector[0][0],arrVector[0][1],0],[arrVector[1][0],arrVector[1][1],0],[0,0,1]]
    print(vector)
    printOct(halfLengthA, pointB, vector, ax)

    ax.scatter(x, y, z, s=1)
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    plt.show()


def printDifObstacle(name1, name2):
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

    ax.scatter(x, y, z, s=1, c='blue')
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

    ax.scatter(x, y, z, s=1, c='red')
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)

    plt.show()

def printObstaclesObb(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    for item in arr:
        printObstacleObbByNotZ(item,ax)
        # c1 = randomcolor()
        # file_obj = open("E:\\file\\点云\\" + item + ".txt", encoding='UTF-8')
        # x = []
        # y = []
        # z = []
        # for line in file_obj.readlines():
        #     line = line.rstrip("\n")
        #     arr = line.split(",")
        #     print(arr)
        #     x.append(float(arr[0]))
        #     y.append(float(arr[1]))
        #     z.append(float(arr[2]))
        #
        # ax.scatter(x, y, z, s=1)

    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    print(plt)
    plt.show()
def printObstacles(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    for item in arr:
        c1 = randomcolor()
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

        ax.scatter(x, y, z, s=1)

    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    print(plt)
    plt.show()


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    colorArr = ['000000', '0000FF', '00008B', '808080', 'FF0000']
    # color = ""
    i = random.randint(0, len(colorArr) - 1)
    print(i)
    color = colorArr[i]
    del colorArr[i]
    # for i in range(6):
    #     color += colorArr[random.randint(0,14)]
    return "#" + color


if __name__ == '__main__':
    # printDifObstacle('2021-01-07-10-28-55点云分离ground','2021-01-07-10-28-55障碍物')
    printObstacle("2021-01-07-10-28-55所有点云")
    # printObstacleObbByNotZTest("2021-01-07-10-04-36obbA")

    # printObstacleObb('2021-01-06-22-11-25点云分离obbE')
    arr = ["2021-01-07-10-28-55点云分离obbA", "2021-01-07-10-28-55点云分离obbB", "2021-01-07-10-28-55点云分离obbD",
           "2021-01-07-10-28-55点云分离obbE", "2021-01-07-10-28-55点云分离球体"]

    # printObstacles(arr)
    # printObstaclesObb(arr)
