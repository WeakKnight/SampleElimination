import matplotlib.pyplot as plt
import math

def weightFunction(d2, dMax, alpha = 8.0):
    return math.pow((1.0 - d2 / dMax), alpha)

def getWeightLimitFraction(inputNum, outputNum, beta = 0.65, gamma = 1.5):
    ratio = float(outputNum) / float(inputNum)
    fraction = (1.0 - math.pow(ratio, gamma)) * beta
    return fraction

def displayVecArray(arr):
    xArr = []
    yArr = []
    for item in arr:
        xArr.append(item[0])
        yArr.append(item[1])
    plt.figure(figsize = (5,5))
    plt.scatter(xArr, yArr)
    plt.show()