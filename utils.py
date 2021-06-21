import matplotlib.pyplot as plt

def display_vec_array(arr):
    xArr = []
    yArr = []
    for item in arr:
        xArr.append(item[0])
        yArr.append(item[1])
    plt.figure(figsize = (5,5))
    plt.scatter(xArr, yArr)
    plt.show()