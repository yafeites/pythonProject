# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as  np


def printOct(halfLength, point, vector, ax):
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

    pointA[4][0] += + halfLength[0] * vector[0][0]
    pointA[4][1] += + halfLength[0] * vector[0][1]
    pointA[4][2] += + halfLength[0] * vector[0][2]
    pointA[4][0] += + halfLength[1] * vector[1][0]
    pointA[4][1] += + halfLength[1] * vector[1][1]
    pointA[4][2] += + halfLength[1] * vector[1][2]
    pointA[4][0] += - halfLength[2] * vector[2][0]
    pointA[4][1] += - halfLength[2] * vector[2][1]
    pointA[4][2] += - halfLength[2] * vector[2][2]

    pointA[5][0] += - halfLength[0] * vector[0][0]
    pointA[5][1] += - halfLength[0] * vector[0][1]
    pointA[5][2] += - halfLength[0] * vector[0][2]
    pointA[5][0] += - halfLength[1] * vector[1][0]
    pointA[5][1] += - halfLength[1] * vector[1][1]
    pointA[5][2] += - halfLength[1] * vector[1][2]
    pointA[5][0] += + halfLength[2] * vector[2][0]
    pointA[5][1] += + halfLength[2] * vector[2][1]
    pointA[5][2] += + halfLength[2] * vector[2][2]

    pointA[6][0] += - halfLength[0] * vector[0][0]
    pointA[6][1] += - halfLength[0] * vector[0][1]
    pointA[6][2] += - halfLength[0] * vector[0][2]
    pointA[6][0] += + halfLength[1] * vector[1][0]
    pointA[6][1] += + halfLength[1] * vector[1][1]
    pointA[6][2] += + halfLength[1] * vector[1][2]
    pointA[6][0] += - halfLength[2] * vector[2][0]
    pointA[6][1] += - halfLength[2] * vector[2][1]
    pointA[6][2] += - halfLength[2] * vector[2][2]

    pointA[7][0] += + halfLength[0] * vector[0][0]
    pointA[7][1] += + halfLength[0] * vector[0][1]
    pointA[7][2] += + halfLength[0] * vector[0][2]
    pointA[7][0] += - halfLength[1] * vector[1][0]
    pointA[7][1] += - halfLength[1] * vector[1][1]
    pointA[7][2] += - halfLength[1] * vector[1][2]
    pointA[7][0] += - halfLength[2] * vector[2][0]
    pointA[7][1] += - halfLength[2] * vector[2][1]
    pointA[7][2] += - halfLength[2] * vector[2][2]

    pointA[3][0] += - halfLength[0] * vector[0][0]
    pointA[3][1] += - halfLength[0] * vector[0][1]
    pointA[3][2] += - halfLength[0] * vector[0][2]
    pointA[3][0] += - halfLength[1] * vector[1][0]
    pointA[3][1] += - halfLength[1] * vector[1][1]
    pointA[3][2] += - halfLength[1] * vector[1][2]
    pointA[3][0] += - halfLength[2] * vector[2][0]
    pointA[3][1] += - halfLength[2] * vector[2][1]
    pointA[3][2] += - halfLength[2] * vector[2][2]

    for i in range(0, len(pointA)):
        for j in range(i):
            if (i != j):
                x = np.linspace(float(pointA[i][0]), float(pointA[j][0]), 2)
                y = np.linspace(float(pointA[i][1]), float(pointA[j][1]), 2)
                z = np.linspace(float(pointA[i][2]), float(pointA[j][2]), 2)
                # print(x,y,z)
                ax.plot(x, y, z, 'r')


def print_tree(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("D:\\file\\" + name + ".txt", encoding='UTF-8')
    halfLengthA = [50, 50, 50]
    halfLengthB = [50, 50, 50]
    pointA = [1300, 0, 800]
    pointB = [1890, 0, 65]
    vector = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    printOct(halfLengthA, pointA, vector, ax)
    printOct(halfLengthB, pointB, vector, ax)
    for line in file_obj.readlines():
        if (line.find("树") != -1):
            continue
        else:
            line = line.rstrip("\n")
            arr1 = line.split(" ")
        # print(arr1)
        s1 = arr1[0].split(",")
        s2 = arr1[1].split(",")
        # print(s2)
        x = np.linspace(float(s1[0]), float(s2[0]), 2)
        y = np.linspace(float(s1[1]), float(s2[1]), 2)
        z = np.linspace(float(s1[2]), float(s2[2]), 2)
        ax.plot(x, y, z, "b");

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_tree('2020-12-21-18-39-57tree')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
