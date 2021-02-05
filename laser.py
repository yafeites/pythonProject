import random

import matplotlib.pyplot as plt
import numpy
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import numpy as  np

from scipy import interpolate
import math


def printOct(halfLength1, point, vector, ax):
    halfLength = [halfLength1[0] + 20, halfLength1[1] + 20, halfLength1[2] + 20]
    print(vector)
    print(point)
    print(halfLength)
    print(8 * halfLength[0] * halfLength[1] * halfLength[2])
    pointA = [[point[0], point[1], point[2]] for i in range(8)]
    # print(vector[0][0])
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
    # print(pointA)

    printLine(pointA, 0, 1, ax)
    printLine(pointA, 4, 1, ax)
    printLine(pointA, 4, 2, ax)
    printLine(pointA, 0, 2, ax)
    printLine(pointA, 5, 1, ax)
    printLine(pointA, 4, 7, ax)
    printLine(pointA, 2, 6, ax)
    printLine(pointA, 0, 3, ax)
    printLine(pointA, 5, 7, ax)
    printLine(pointA, 5, 3, ax)
    printLine(pointA, 7, 6, ax)
    printLine(pointA, 6, 3, ax)


def printLine(pointA, i, j, ax):
    x = np.linspace(float(pointA[i][0]), float(pointA[j][0]), 2)
    y = np.linspace(float(pointA[i][1]), float(pointA[j][1]), 2)
    z = np.linspace(float(pointA[i][2]), float(pointA[j][2]), 2)
    ax.plot(x, y, z, 'red')


def printObstacle(name):
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name + ".txt", encoding='UTF-8')

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

    ax.scatter(x, y, z, s=1)

    plt.xlim(690, 3550)
    # plt.xlim(1200, 1500)
    plt.ylim(-1200, 1200)

    plt.show()


def printObstacleObb(name):
    arr1 = [[1, 2, 3]]
    arr2 = [[4, 0, 3]]
    # print(np.dot(np.transpose(arr1) ,arr2))
    print(np.dot(arr1, np.transpose(arr2)))
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name + ".txt", encoding='UTF-8')

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

    avg = np.array([0, 0, 0])
    # print(avg)
    # avg[0] = 0
    # avg[1] = 0
    # avg[2] = 0
    for i in range(size):
        avg[0] += x[i]
        avg[1] += y[i]
        avg[2] += z[i]
    avg[0] /= size
    avg[1] /= size
    avg[2] /= size
    arr = numpy.linalg.eig(numpy.cov(np.transpose(points)))
    print(arr)
    # print(numpy.linalg.eig(numpy.cov(points)))
    # for i in range(size):
    #     for j in range(size):
    #         num_list[i][j] =
    maxL = -100000
    maxW = -100000
    maxH = -100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL = max(maxL, np.dot(point, np.transpose(arr[1][0])))
        minL = min(minL, np.dot(point, np.transpose(arr[1][0])))
        maxW = max(maxW, np.dot(point, np.transpose(arr[1][1])))
        minW = min(minW, np.dot(point, np.transpose(arr[1][1])))
        maxH = max(maxH, np.dot(point, np.transpose(arr[1][2])))
        minH = min(minH, np.dot(point, np.transpose(arr[1][2])))
        # print(minL)
    tr = np.transpose(arr[1])
    print(tr)
    pointA = [(maxL + minL) / 2, (maxW + minW) / 2, (maxH + minH) / 2]

    X = np.dot(pointA, np.transpose(tr[0]))
    Y = np.dot(pointA, np.transpose(tr[1]))
    Z = np.dot(pointA, np.transpose(tr[2]))
    pointB = [X, Y, Z]
    print(X, Y, Z)

    halfLengthA = [(maxL - minL) / 2, (maxW - minW) / 2, (maxH - minH) / 2]

    printOct(halfLengthA, pointB, arr[1], ax)

    ax.scatter(x, y, z, s=1)
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    plt.show()


def printObstacleObbByNotZ(name, ax):
    print(name)
    # arr1 = [[1, 2, 3]]
    # arr2 = [[4, 0, 3]]
    # print(np.dot(np.transpose(arr1) ,arr2))
    # print(np.dot(arr1 ,np.transpose(arr2)))

    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name + ".txt", encoding='UTF-8')

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
    avg = np.array([0, 0])
    # print(avg)
    # avg[0] = 0
    # avg[1] = 0
    # avg[2] = 0
    for i in range(size):
        avg[0] += x[i]
        avg[1] += y[i]
        # avg[2] += z[i]
    avg[0] /= size
    avg[1] /= size
    # avg[2]/=size
    arr = numpy.linalg.eig(numpy.cov(np.transpose(pointsNotZ)))
    # print(arr)
    # print(numpy.linalg.eig(numpy.cov(points)))
    maxL = -100000
    maxW = -100000
    maxH = -100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL = max(maxL, np.dot(point[:2], np.transpose(arr[1][0])))
        minL = min(minL, np.dot(point[:2], np.transpose(arr[1][0])))
        maxW = max(maxW, np.dot(point[:2], np.transpose(arr[1][1])))
        minW = min(minW, np.dot(point[:2], np.transpose(arr[1][1])))
        maxH = max(maxH, point[2])
        minH = min(minH, point[2])
        # print(minL)
    tr = np.transpose(arr[1])
    # print(tr)
    pointA = [(maxL + minL) / 2, (maxW + minW) / 2, (maxH + minH) / 2]

    X = np.dot(pointA[:2], np.transpose(tr[0]))
    Y = np.dot(pointA[:2], np.transpose(tr[1]))
    Z = pointA[2]
    pointB = [X, Y, Z]
    # print(X,Y,Z)
    arrVector = arr[1]
    # print(arrVector)
    halfLengthA = [(maxL - minL) / 2, (maxW - minW) / 2, (maxH - minH) / 2]
    vector = [[arrVector[0][0], arrVector[0][1], 0], [arrVector[1][0], arrVector[1][1], 0], [0, 0, 1]]
    # print(vector)
    printOct(halfLengthA, pointB, vector, ax)

    ax.scatter(x, y, z, s=1)
    # plt.xlim(1000, 2500)
    # plt.ylim(-1000, 1000)
    # plt.show()


def printObstacleObbSphere(name, ax):
    print(name)

    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name + ".txt", encoding='UTF-8')

    x = []
    y = []
    z = []
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
    size = len(x)
    points = numpy.zeros((size, 3))

    for i in range(size):
        points[i][0] = x[i]
        points[i][1] = y[i]
        points[i][2] = z[i]

    maxL = -100000
    maxW = -100000
    maxH = -100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL = max(maxL, point[0])
        minL = min(minL, point[0])
        maxW = max(maxW, point[1])
        minW = min(minW, point[1])
        maxH = max(maxH, point[2])
        minH = min(minH, point[2])

    pointA = [(maxL + minL) / 2, (maxW + minW) / 2, (maxH + minH) / 2]

    halfLengthA = [(maxL - minL) / 2, (maxW - minW) / 2, (maxH - minH) / 2]
    # vector = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # printOct(halfLengthA, pointA, vector, ax)
    r = 0
    for point in points:
        r = max(r,
                math.sqrt(pow(pointA[0] - point[0], 2) + pow(pointA[1] - point[1], 2) + pow(pointA[2] - point[2], 2)))

    # s = sphere(pos=(pointA))
    # s.radius=r
    # ax.plot(s)
    center = pointA
    radius = r
    print(2 * 3.14 * r * r)
    # data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x1 = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y1 = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z1 = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    # plot

    # surface plot

    # wire frame
    ax.plot_wireframe(x1, y1, z1, rstride=10, cstride=10)
    ax.scatter(x, y, z, s=1)
    # plt.xlim(1000, 2500)
    # plt.ylim(-1000, 1000)
    # plt.show()


def printObstacleObbSquare(name, ax):
    print(name)

    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name + ".txt", encoding='UTF-8')

    x = []
    y = []
    z = []
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
    size = len(x)
    points = numpy.zeros((size, 3))

    for i in range(size):
        points[i][0] = x[i]
        points[i][1] = y[i]
        points[i][2] = z[i]

    maxL = -100000
    maxW = -100000
    maxH = -100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL = max(maxL, point[0])
        minL = min(minL, point[0])
        maxW = max(maxW, point[1])
        minW = min(minW, point[1])
        maxH = max(maxH, point[2])
        minH = min(minH, point[2])

    pointA = [(maxL + minL) / 2, (maxW + minW) / 2, (maxH + minH) / 2]

    halfLengthA = [(maxL - minL) / 2, (maxW - minW) / 2, (maxH - minH) / 2]
    vector = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    printOct(halfLengthA, pointA, vector, ax)
    ax.plot
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
    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name + ".txt", encoding='UTF-8')

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
    avg = np.array([0, 0])
    # print(avg)
    # avg[0] = 0
    # avg[1] = 0
    # avg[2] = 0
    for i in range(size):
        avg[0] += x[i]
        avg[1] += y[i]
        # avg[2] += z[i]
    avg[0] /= size
    avg[1] /= size
    # avg[2]/=size
    arr = numpy.linalg.eig(numpy.cov(np.transpose(pointsNotZ)))
    print(arr)
    # print(numpy.linalg.eig(numpy.cov(points)))
    maxL = -100000
    maxW = -100000
    maxH = -100000
    minL = 100000
    minW = 100000
    minH = 100000

    for point in points:
        maxL = max(maxL, np.dot(point[:2], np.transpose(arr[1][0])))
        minL = min(minL, np.dot(point[:2], np.transpose(arr[1][0])))
        maxW = max(maxW, np.dot(point[:2], np.transpose(arr[1][1])))
        minW = min(minW, np.dot(point[:2], np.transpose(arr[1][1])))
        maxH = max(maxH, point[2])
        minH = min(minH, point[2])
        # print(minL)
    tr = np.transpose(arr[1])
    print(tr)
    pointA = [(maxL + minL) / 2, (maxW + minW) / 2, (maxH + minH) / 2]

    X = np.dot(pointA[:2], np.transpose(tr[0]))
    Y = np.dot(pointA[:2], np.transpose(tr[1]))
    Z = pointA[2]
    pointB = [X, Y, Z]
    print(X, Y, Z)
    arrVector = arr[1]
    print(arrVector)
    halfLengthA = [(maxL - minL) / 2, (maxW - minW) / 2, (maxH - minH) / 2]
    vector = [[arrVector[0][0], arrVector[0][1], 0], [arrVector[1][0], arrVector[1][1], 0], [0, 0, 1]]
    print(vector)
    printOct(halfLengthA, pointB, vector, ax)

    ax.scatter(x, y, z, s=1)
    plt.xlim(1000, 2500)
    plt.ylim(-1000, 1000)
    plt.show()


def printDifObstacle(name1, name2):
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name1 + ".txt", encoding='UTF-8')
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
    file_obj = open("E:\\graduateDesignTxt\\点云\\" + name2 + ".txt", encoding='UTF-8')
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
    plt.xlim(690, 3550)
    plt.ylim(-1200, 1200)

    plt.show()


def printObstaclesSquare(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    for item in arr:
        printObstacleObbSquare(item, ax)

    plt.xlim(690, 3550)
    plt.ylim(-1200, 1200)
    print(plt)
    plt.show()


def printObstaclesSphere(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    for item in arr:
        printObstacleObbSphere(item, ax)

    plt.xlim(690, 3550)
    plt.ylim(-1200, 1200)
    print(plt)
    plt.show()


def printObstaclesObb(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    for item in arr:
        printObstacleObbByNotZ(item, ax)
        # c1 = randomcolor()
        # file_obj = open("E:\\graduateDesignTxt\\点云\\" + item + ".txt", encoding='UTF-8')
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

    plt.xlim(690, 3550)
    plt.ylim(-1200, 1200)
    print(plt)
    plt.show()

    def printObstaclessquare(arr):
        fig = plt.figure()
        ax = Axes3D(fig)
        for item in arr:
            printObstacleObbByNotZ(item, ax)
            # c1 = randomcolor()
            # file_obj = open("E:\\graduateDesignTxt\\点云\\" + item + ".txt", encoding='UTF-8')
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

        plt.xlim(690, 3550)
        plt.ylim(-1200, 1200)
        print(plt)
        plt.show()


def printObstacles(arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    i = 0
    for item in (arr):
        print(arr)

        # print(index,item)
        # c1 = randomcolor()
        file_obj = open("E:\\graduateDesignTxt\\点云\\" + item + ".txt", encoding='UTF-8')
        x = []
        y = []
        z = []
        for line in file_obj.readlines():
            line = line.rstrip("\n")
            arrs = line.split(",")
            # print(arr)
            x.append(float(arrs[0]))
            y.append(float(arrs[1]))
            z.append(float(arrs[2]))
        if (i == len(arr) - 1):
            ax.scatter(x, y, z, s=1, c='black')
            i += 1
            continue
        ax.scatter(x, y, z, s=1)
        i += 1

    plt.xlim(690, 2500)
    plt.ylim(-1000, 1000)
    # print(plt)
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
    # printDifObstacle('20210127 114718地面点', '20210127 114718分离地面')
    printObstacle('2021-01-28-14-48-08所有点云')
    # printObstacleObbByNotZTest("2021-01-07-10-04-36obbA")

    # printObstacleObb('2021-01-06-22-11-25点云分离obbE')
    arr = ["2021-01-18-21-56-10点云分离Elobstacle1", "2021-01-18-21-56-10点云分离Elobstacle2",
           "2021-01-18-21-56-10点云分离Elobstacle3",
           "2021-01-18-21-56-10点云分离Elobstacle4", "2021-01-18-21-56-10点云分离El球体"]
    # 实验1
    # arr=['2021-01-18-12-04-37点云分离E1obstacle1']

    # 实验2
    # arr=['2021-01-18-16-09-34点云分离E2obstacle1','2021-01-18-16-09-34点云分离E2obstacle2']

    # 实验3
    # arr=['2021-01-18-18-57-14点云分离E3obstacle1','2021-01-18-18-57-14点云分离E3obstacle2','2021-01-18-18-57-14点云分离E3obstacle3']
    # 激光雷达实验角度1
    # arr=['2021-01-19-16-28-51点云分离Elobstacle1','2021-01-19-16-28-51点云分离Elobstacle2',
    #      '2021-01-19-16-28-51点云分离Elobstacle3','2021-01-19-16-28-51点云分离Elobstacle4'
    #     ,'2021-01-19-16-28-51点云分离El球体']
    # 激光雷达实验角度1.5
    arr = ['2021-01-19-16-31-37点云分离Elobstacle1', '2021-01-19-16-31-37点云分离Elobstacle2',
           '2021-01-19-16-31-37点云分离Elobstacle3', '2021-01-19-16-31-37点云分离Elobstacle4',
           '2021-01-19-16-31-37点云分离El球体']
    # 激光雷达实验角度2

    arr = ['2021-01-19-16-33-31点云分离Elobstacle1', '2021-01-19-16-33-31点云分离Elobstacle2',
           '2021-01-19-16-33-31点云分离Elobstacle3', '2021-01-19-16-33-31点云分离Elobstacle4',
           '2021-01-19-16-33-31点云分离El球体']
    # 激光雷达实验角度3
    arr = ['2021-01-19-16-35-25点云分离Elobstacle1', '2021-01-19-16-35-25点云分离Elobstacle2',
           '2021-01-19-16-35-25点云分离Elobstacle3', '2021-01-19-16-35-25点云分离Elobstacle4',
           '2021-01-19-16-35-25点云分离El球体']

    # 欧式聚类

    # 60+15
    arr = ['2021-01-28-11-56-50欧式聚类0', '2021-01-28-11-56-50欧式聚类1', '2021-01-28-11-56-50欧式聚类2',
           '2021-01-28-11-56-50欧式聚类3',
           '2021-01-28-11-56-50欧式聚类4','2021-01-28-11-56-50欧式聚类5']
    # 40+25
    # arr = ['2021-01-28-11-53-57欧式聚类0', '2021-01-28-11-53-57欧式聚类1', '2021-01-28-11-53-57欧式聚类2',
    #        '2021-01-28-11-53-57欧式聚类3', '2021-01-28-11-53-57欧式聚类4']
    # 60 + 25
    # arr = ['2021-01-28-11-49-03欧式聚类0', '2021-01-28-11-49-03欧式聚类2', '2021-01-28-11-49-03欧式聚类1',
    #        '2021-01-28-11-49-03欧式聚类3'
    #     , '2021-01-28-11-49-03欧式聚类4']

    # arr = ['2021-01-22-10-54-23欧式聚类5']
    # 第三章例图
# arr = ['2021-01-27-11-41-01点云分离Elobstacle1', '2021-01-27-11-41-01点云分离Elobstacle2']
# printObstacles(arr)
# 1度obb
# arr=['2021-01-26-21-43-04欧式聚类0','2021-01-26-21-43-04欧式聚类1','2021-01-26-21-43-04欧式聚类2','2021-01-26-21-43-04欧式聚类3'
#     ,'2021-01-26-22-03-16Elobstacle2']
# 1.5度obb
# arr = ['2021-01-26-22-30-16点云分离Elobstacle1', '2021-01-26-22-30-16点云分离Elobstacle2', '2021-01-26-22-30-16点云分离Elobstacle3',
#        '2021-01-26-22-30-16点云分离Elobstacle4', '2021-01-26-22-30-16点云分离El球体']
# 2度obb
arr = ['2021-01-26-22-37-07点云分离Elobstacle1', '2021-01-26-22-37-07点云分离Elobstacle2', '2021-01-26-22-37-07点云分离Elobstacle3',
       '2021-01-26-22-37-07点云分离Elobstacle4', '2021-01-26-22-37-07点云分离El球体']
# 3度obb
arr = ['2021-01-26-22-42-35点云分离Elobstacle1', '2021-01-26-22-42-35点云分离Elobstacle2', '2021-01-26-22-42-35点云分离Elobstacle3',
       '2021-01-26-22-42-35点云分离Elobstacle4', '2021-01-26-22-42-35点云分离El球体']
# 激光雷达实验1
arr = ['2021-01-28-14-17-43点云分离E1obstacle1']
# 激光雷达实验2
arr = ['2021-01-28-14-33-00E2obstacle1', '2021-01-28-14-33-07E2obstacle2']
# 激光雷达实验3
arr = ['2021-01-27-10-40-17E3obstacle1', '2021-01-27-10-44-07E3obstacle2', '2021-01-27-10-44-32E3obstacle3']
# 第三章例图
# arr = ['2021-01-27-11-41-01点云分离Elobstacle1', '2021-01-27-11-41-01点云分离Elobstacle2']
# 60 + 25 包络测试
# arr = ['2021-01-28-11-49-03欧式聚类0', '2021-01-28-11-49-03欧式聚类2', '2021-01-28-11-49-03欧式聚类1',
#        '2021-01-28-11-49-03欧式聚类3'
#     ]
arr=['2021-01-26-22-40-21Elobstacle1']
printObstaclesSquare(arr)
# printObstaclesSphere(arr)
printObstaclesObb(arr)
