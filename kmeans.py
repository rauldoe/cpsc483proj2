import numpy as np
import matplotlib.pyplot as plt

def distanceBetweenPointsAndCentroids(X, centroids, K):

    num_points = len(X)
    print(f'Total number of points in the dataset, ie. X is {num_points}')

    for i in range(K):
        for j in range(num_points):
            dist[i, j] = np.linalg.norm(centroids[i] - X[j])
    return dist

def getGrp(X, dist):
    grp0 = []
    grp1 = []
    for i in range(len(dist[0])):
        if (dist[0][i] < dist[1][i]):
            grp0.append(X[i])
        else:
            grp1.append(X[i])
    return [grp0, grp1]


def getCentroid(grp):
    totalX = 0.0
    totalY = 0.0
    for i in range(len(grp)):
        totalX += grp[i][0]
        totalY += grp[i][1]

    return [totalX/len(grp), totalY/len(grp)]

def isEqual(arr0, arr1):
    return (arr0[0] == arr1[0]) and (arr0[1] == arr1[1])

def isEqualGrpList(newGrpList, oldGrpList):
    if (len(newGrpList[0]) != len(oldGrpList[0])):
        return False
    if (len(newGrpList[1]) != len(oldGrpList[1])):
        return False

    for i in range(len(newGrpList[0])):
        if (not isEqual(newGrpList[0][i], oldGrpList[0][i])):
            return False

    for i in range(len(newGrpList[1])):
        if (not isEqual(newGrpList[1][i], oldGrpList[1][i])):
            return False
    return True

X = np.array([[1,1], [2,1], [4,3], [5,4]])

print(f'Printing data in X... {X} Dimensions of X {X.shape}')

K = 2

centroids = np.array([[1,1], [2,1]])

# plt.plot(X[:,0], X[:1], 'go')
# plt.plot(initial_centroids[:,0], initial_centroids[:1], 'rx')

dist = np.zeros((K, len(X)))

doStop = False

oldGrpList = None
while doStop == False:
    dist = distanceBetweenPointsAndCentroids(X, centroids, K)

    grpList = getGrp(X, dist)

    c0 = getCentroid(grpList[0])
    c1 = getCentroid(grpList[1])

    centroids[0] = c0
    centroids[1] = c1

    if (oldGrpList is not None and isEqualGrpList(grpList, oldGrpList)):
        doStop = True
    else:
        oldGrpList = grpList

print(f'c0: {centroids[0]}, c1: {centroids[1]}')
print(oldGrpList)
