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
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
import math

np.random.seed(0)

alpha = 8

inputNum = 250
outputNum = 50
initialSamples = np.random.random((inputNum, 2))  # 5 points in 2 dimensions

utils.displayVecArray(initialSamples)

tree = spatial.KDTree(initialSamples)
weights = np.zeros(inputNum)

domainArea = 1.0 * 1.0 / outputNum
domainDimension = 2

rMax = math.sqrt(domainArea / (2 * math.sqrt(3)))
for i in range(inputNum):
    neighbourIndices = tree.query_ball_point(initialSamples[i], r = 2.0 * rMax)
    for neighbourIndex in neighbourIndices:
        squaredDistance = pairwise_distances([initialSamples[i]], [initialSamples[neighbourIndex]]).flatten()[0]
        weights[i] += utils.weightFunction(squaredDistance, 2.0 * rMax)

heap = MaxHeap()
heap.SetData(weights)
heap.Build()

outputSamples = []
remainSampleNum = inputNum
while remainSampleNum > outputNum:
    index = int(heap.GetTopItemID())
    heap.Pop()

    neighbourIndices = tree.query_ball_point(initialSamples[index], r = 2.0 * rMax)
    for neighbourIndex in neighbourIndices:
        if neighbourIndex != index:
            squaredDistance = pairwise_distances([initialSamples[index]], [initialSamples[neighbourIndex]]).flatten()[0]
            weights[neighbourIndex] -= utils.weightFunction(squaredDistance, 2.0 * rMax)
            heap.MoveItemDown(neighbourIndex)
    remainSampleNum -= 1

for i in range(outputNum):
    outputSamples.append(initialSamples[int(heap.GetIDFromHeap(i))])

utils.displayVecArray(outputSamples)


# %%
import numpy as np
import utils
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
import math

np.random.seed(0)
NearestNeighborCount = 5
IterationNum = 20
outputNum = 50
initialSamples = np.random.random((outputNum, 2))  # 5 points in 2 dimensions
utils.displayVecArray(initialSamples)
outputSamples = np.copy(initialSamples)
for iterationIndex in range(IterationNum):
    tree = spatial.KDTree(outputSamples)
    for i in range(len(outputSamples)):
        sample = outputSamples[i]
        _, ii = tree.query(sample, NearestNeighborCount + 1)
        squaredRMax = 0
        for neighbourIndex in ii:
            squaredDistance = pairwise_distances([sample], [outputSamples[int(neighbourIndex)]]).flatten()[0]
            if squaredDistance > squaredRMax:
                squaredRMax = squaredDistance
        deltaX = np.array([0.0, 0.0])
        for neighbourIndex in ii:
            squaredDistance = pairwise_distances([sample], [outputSamples[neighbourIndex]]).flatten()[0]
            if squaredDistance < squaredRMax:
                movement = (1.0 / float(NearestNeighborCount)) * (sample - outputSamples[neighbourIndex]) * (math.sqrt(squaredRMax) / (math.sqrt(squaredDistance) + 1e-6) - 1.0)
                deltaX += movement
        sample += deltaX
        outputSamples[i] = sample
utils.displayVecArray(outputSamples)

# %%
import numpy as np
import utils
from max_heap import MaxHeap
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
import math

np.random.seed(0)

alpha = 8

inputNum = 250
outputNum = 70
nearestNeighborCount = 8
initialSamples = np.random.random((inputNum, 2))  # 5 points in 2 dimensions

utils.displayVecArray(initialSamples)

tree = spatial.KDTree(initialSamples)
# Compute RMax List
rMaxs = np.zeros(inputNum)
sr = np.zeros(inputNum)
ratio = float(inputNum) / float(outputNum)
for i in range(len(initialSamples)):
    initialSample = initialSamples[i]
    dd, ii = tree.query(initialSample, nearestNeighborCount)
    radius = dd[len(dd) - 1]
    sampleArea = ratio * (math.pi * radius * radius) / float(len(dd))
    rMaxs[i] = math.sqrt(sampleArea / (2.0 * math.sqrt(3.0)))
    sr[i] = 2.0 * rMaxs[i]

weights = np.zeros(inputNum)

for index in range(inputNum):
    dMax = 2.0 * rMaxs[index]
    neighbourIndices = tree.query_ball_point(initialSamples[index], r = dMax)
    for i in neighbourIndices:
        if i >= inputNum:
            continue
        if i != index:
            squaredDistance = pairwise_distances([initialSamples[index]], [initialSamples[i]]).flatten()[0]
            weights[index] += utils.weightFunction(squaredDistance, dMax)
            # Why
            if sr[i] < dMax:
                sr[i] = dMax

heap = MaxHeap()
heap.SetData(weights)
heap.Build()

outputSamples = []
remainSampleNum = inputNum
while remainSampleNum > outputNum:
    index = int(heap.GetTopItemID())
    heap.Pop()

    neighbourIndices = tree.query_ball_point(initialSamples[index], r = sr[index])
    for i in neighbourIndices:
        if i >= inputNum:
            continue
        if i != index:
            neighbourDMax = 2.0 * rMaxs[i]
            squaredDistance = pairwise_distances([initialSamples[index]], [initialSamples[i]]).flatten()[0]
            if (neighbourDMax * neighbourDMax) > squaredDistance:
                weights[i] -= utils.weightFunction(squaredDistance, neighbourDMax)
                heap.MoveItemDown(i)
    remainSampleNum -= 1

for i in range(outputNum):
    outputSamples.append(initialSamples[int(heap.GetIDFromHeap(i))])

utils.displayVecArray(outputSamples)
# %%
