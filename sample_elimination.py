# from sklearn.neighbors import KDTree
# tree = KDTree(X)
# nearest_dist, nearest_ind = tree.query(X, k=2)  # k=2 nearest neighbors where k1 = identity
# print(X)
# print(nearest_dist[:, 1])    # drop id; assumes sorted -> see args!
# print(nearest_ind[:, 1])     # drop id 

#%%
import numpy as np
import utils
from max_heap import MaxHeap
def display_vec_array(arr):
    xArr = []
    yArr = []
    for item in arr:
        xArr.append(item[0])
        yArr.append(item[1])
    plt.figure(figsize = (5,5))
    plt.scatter(xArr, yArr)
    plt.show()

np.random.seed(0)
X = np.random.random((10, 1))  # 5 points in 2 dimensions
heap = MaxHeap(X)
# utils.display_vec_array(X)

# %%
