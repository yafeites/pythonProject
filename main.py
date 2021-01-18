import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as  np
from scipy import interpolate
import math


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
    # for i in range(0, len(pointA)):
    #     for j in range(i):
    #         if (i != j):
    #             x = np.linspace(float(pointA[i][0]), float(pointA[j][0]), 2)
    #             y = np.linspace(float(pointA[i][1]), float(pointA[j][1]), 2)
    #             z = np.linspace(float(pointA[i][2]), float(pointA[j][2]), 2)
    #             # print(x,y,z)
    #             ax.plot(x, y, z, 'black')
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
             # print(x,y,z)
            ax.plot(x, y, z, 'black')
def print_Dif_tree(name, name1):
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\file\\" + name + ".txt", encoding='UTF-8')
    printObs(ax)

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
        line1, = ax.plot(x, y, z, 'b')

    file_obj = open("E:\\file\\" + name1 + ".txt", encoding='UTF-8')

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
        line2, = ax.plot(x, y, z, 'r')
    plt.rcParams['font.sans-serif'] = ['SimHei']  ###解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(handles=[line1, line2], labels=['rrt', '联合算法'], loc='best')
    plt.show()




def print_tree(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    fig = plt.figure()
    ax = Axes3D(fig)

    file_obj = open("E:\\file\\" + name + ".txt", encoding='UTF-8')
    printObs(ax)

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
        ax.plot(x, y, z)
    plt.xlim(1000,3000)
    # plt.ylim(-1200,1200)
    plt.show()



def fit(name):
    fig = plt.figure()
    ax = Axes3D(fig)
    file_obj = open("E:\\file\\" + name + ".txt", encoding='UTF-8')
    x = []
    y = []
    z = []
    i = 0
    for line in file_obj.readlines():
        line = line.rstrip("\n")
        arr = line.split(",")
        # print(arr)
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
        # i=i+1
    print(x)
    print(y)
    print(z)
    printObs(ax)
    # print(np.array(x))
    tck, u = interpolate.splprep([np.array(x), np.array(y), np.array(z)], k=3)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)
    print(out)
    plt.rcParams['font.sans-serif'] = ['SimHei']  ###解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False
    # ax.plot(x, y, z, c='red', label='原始曲线')
    ax.plot(out[0], out[1], out[2])
    plt.legend()

    plt.show()


def printObs(ax):
    halfLengthA = [50, 50,50]
    halfLengthB = [50, 50, 50]
    halfLengthC = [50, 50,50]
    # 实验1
    vectorE1 = [[0.9999917896676187, 0.004052233625149753, 0], [-0.004052233625149753, 0.9999917896676187, 0], [0, 0, 1]]
    halfLengthOb1=[168.9698656966724, 85.44732829925721, 129.9982925488161]
    pointOb1=[ 1878.8424834265804, -0.0043320454585780155, -160.0017074511839]

    printOct(halfLengthOb1, pointOb1, vectorE1, ax)

    # 实验2
    # halfLengthOb1=[150,110,55]
    # pointOb1=[1890,0,-220]
    # 实验3
    # halfLengthOb1=[150,55,110]
    # halfLengthOb2=[140,600,65]
    # halfLengthOb3=[150,105,175]
    #
    #
    # pointOb1=[1890.0,0.0,-170]
    # pointOb2=[1650.0,600,330]
    # pointOb3=[1650.0,-355,-95]
    # pointB = [1300, 0, 800]
    # pointA = [1890, 0, 65]
    # pointC = [1500, 50, 300]
    # pointD = [1500, 0, 65.0]
    # pointE = [1700.0, -50, 65.0]
    pointA = [1890, 0, 65]
    pointB = [1300, 0, 800]
    pointC = [1500, 50, 300]
    pointD = [1500, 0, 65.0]
    pointE = [1700.0, -50, 65.0]
    vector = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # printOct(halfLengthOb1, pointOb1, vector, ax)
    # printOct(halfLengthOb2, pointOb2, vector, ax)
    # printOct(halfLengthOb3, pointOb3, vector, ax)



    # printOct(halfLengthA, pointA, vector, ax)
    # printOct(halfLengthB, pointB, vector, ax)
    # printOct(halfLengthC, pointC, vector, ax)

    # printOct(halfLengthC, pointD, vector, ax)
    # printOct(halfLengthC, pointE, vector, ax)


# Press the green button in the gutter to run the script.
def printLength(name):
    fig = plt.figure()
    ax = Axes3D(fig)
    file_obj = open("E:\\file\\" + name + ".txt", encoding='UTF-8')
    printObs(ax)
    dis = 0
    for line in file_obj.readlines():
        if (line.find("树") != -1):
            continue
        else:
            line = line.rstrip("\n")
            arr1 = line.split(" ")
        # print(arr1)
        s1 = arr1[0].split(",")
        s2 = arr1[1].split(",")
        dis += math.sqrt(math.pow(float(s1[0]) - float(s2[0]), 2) + math.pow(float(s1[1]) - float(s2[1]), 2) + math.pow(
            float(s1[2]) - float(s2[2]), 2))

        # # print(s2)
        # x = np.linspace(float(s1[0]), float(s2[0]), 2)
        # y = np.linspace(float(s1[1]), float(s2[1]), 2)
        # z = np.linspace(float(s1[2]), float(s2[2]), 2)
        # ax.plot(x, y, z)
    print(dis)




if __name__ == '__main__':
    # print_tree('2021-01-18-12-32-08optnode')
    # printLength('2021-01-14-17-20-52optnode')


    # print_Dif_tree('2020-12-28-18-49-10optnode','五障碍物\\2020-12-24-18-43-03optnode')

    fit('2021-01-18-12-32-08point')