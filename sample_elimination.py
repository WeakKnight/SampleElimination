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

inputNum = 250
outputNum = 50
initialSamples = np.concatenate((np.random.random((int(inputNum * 0.4), 2)), np.array([0.25, 0.25]) + 0.5 * np.random.random((int(inputNum * 0.6), 2))), axis=0)
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

#%%
import numpy as np
import utils
from max_heap import MaxHeap
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
import math

def se(initialSamples, outputNum, domainArea):
    inputNum = len(initialSamples)
    tree = spatial.KDTree(initialSamples)
    weights = np.zeros(inputNum)
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
    return outputSamples

np.random.seed(0)

# inputNum = 250
# outputNum = 50
# initialSamples = np.concatenate((np.random.random((int(inputNum * 0.4), 2)), np.array([0.25, 0.25]) + 0.5 * np.random.random((int(inputNum * 0.6), 2))), axis=0)

inputNumA = 100
outputNumA = 20
initialSamplesA = np.random.random((inputNumA, 2))
# utils.displayVecArray(initialSamplesA)

domainAreaA = 1.0 * 1.0 / outputNumA
outputSamplesA = se(initialSamplesA, outputNumA, domainAreaA)

# utils.displayVecArray(outputSamplesA)

inputNumB = 150
outputNumB = 30
initialSamplesB = np.array([0.25, 0.25]) + 0.5 * np.random.random((inputNumB, 2))
# utils.displayVecArray(initialSamplesB)
domainAreaB = 0.5 * 0.5 / outputNumB
outputSamplesB = se(initialSamplesB, outputNumB, domainAreaB)

# utils.displayVecArray(outputSamplesB)

outputSamplesC = np.concatenate((outputSamplesA, outputSamplesB), axis=0)
utils.displayVecArray(outputSamplesC)

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
initialSamples = np.concatenate((np.random.random((int(outputNum * 0.4), 2)), np.array([0.25, 0.25]) + 0.5 * np.random.random((int(outputNum * 0.6), 2))), axis=0)
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

inputNum = 250
outputNum = 50
nearestNeighborCount = 120
# initialSamples = np.random.random((inputNum * 0.5, 2))  # 5 points in 2 dimensions
initialSamples = np.concatenate((np.random.random((int(inputNum * 0.4), 2)), np.array([0.25, 0.25]) + 0.5 * np.random.random((int(inputNum * 0.6), 2))), axis=0)
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

# gtOutput = np.array([[0.677817,0.270008],[0.602763,0.544883],[0.423655,0.645894],[0.437587,0.891773],[0.963663,0.383442],[0.791725,0.528895],[0.568045,0.925597],[0.0710361,0.0871293],[0.0202184,0.83262],[0.778157,0.870012],[0.978618,0.799159],[0.461479,0.780529],[0.118274,0.639921],[0.143353,0.944669],[0.521848,0.414662],[0.264556,0.774234],[0.45615,0.568434],[0.0187898,0.617635],[0.612096,0.616934],[0.943748,0.68182],[0.359508,0.437032],[0.697631,0.0602255],[0.666767,0.670638],[0.210383,0.128926],[0.315428,0.363711],[0.570197,0.438602],[0.988374,0.102045],[0.208877,0.16131],[0.653108,0.253292],[0.466311,0.244426],[0.15897,0.110375],[0.65633,0.138183],[0.196582,0.368725],[0.820993,0.0971013],[0.837945,0.0960984],[0.976459,0.468651],[0.976761,0.604846],[0.739264,0.0391878],[0.282807,0.120197],[0.29614,0.118728],[0.317983,0.414263],[0.0641475,0.692472],[0.566601,0.26539],[0.523248,0.0939405],[0.575947,0.929296],[0.318569,0.66741],[0.131798,0.716327],[0.289406,0.183191],[0.586513,0.0201076],[0.82894,0.00469548]])
# utils.displayVecArray(gtOutput)
# %%
