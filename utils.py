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
    fig = plt.figure(figsize = (5,5))
    fig.patch.set_facecolor('#1e1e1e')
    ax = fig.subplots()
    ax.xaxis.label.set_color('#cccccc')
    ax.tick_params(axis='x', colors='#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.tick_params(axis='y', colors='#cccccc')
    
    ax.scatter(xArr, yArr)
    
    plt.show()