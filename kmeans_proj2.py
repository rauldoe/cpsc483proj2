import numpy as np
# import matplotlib.pyplot as plt
import scipy.io

def distanceBetweenPointsAndCentroids(X, centroids, K, dist):

    num_points = len(X)
    # print(f'Total number of points in the dataset, ie. X is {num_points}')

    for indexOfCentroids in range(K):
        for indexOfPoints in range(num_points):
            dist[indexOfCentroids, indexOfPoints] = np.linalg.norm(centroids[indexOfCentroids] - X[indexOfPoints])
    return dist

def getCentroidMemberships(X, dist, K):
    centroidMemberships = []
    for _ in range(K):
        centroidMemberships.append([])

    for indexOPoints in range(len(dist[0])):
        minDistanceBetweenCentroids = dist[0][indexOPoints]
        minIndexOfCentroids = 0
        
        for indexOfCentroids in range(1,K):
            if (dist[indexOfCentroids, indexOPoints] < minDistanceBetweenCentroids):
                minDistanceBetweenCentroids = dist[indexOfCentroids][indexOPoints]
                minIndexOfCentroids = indexOfCentroids

        centroidMemberships[minIndexOfCentroids].append(X[indexOPoints])

    return centroidMemberships

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

datafile = 'kmeansdata.mat'
datafileNew = 'kmeansdata_new.mat'
points = scipy.io.loadmat(datafile)

# scipy.io.savemat(datafileNew, points)

# X = np.array([[1,1], [2,1], [4,3], [5,4]])
X = points['X']

# print(f'Printing data in X... {X} Dimensions of X {X.shape}')

K = 3

centroids = np.array([[3,3], [6,2], [8,5]])

# plt.plot(X[:,0], X[:1], 'go')
# plt.plot(initial_centroids[:,0], initial_centroids[:1], 'rx')

numberOfPoints = len(X)
dist = np.zeros((K, numberOfPoints))

doStop = False

oldGrpList = None
while doStop == False:
    dist = distanceBetweenPointsAndCentroids(X, centroids, K, dist)

    grpList = getCentroidMemberships(X, dist, K)

    # c0 = getCentroid(grpList[0])
    # c1 = getCentroid(grpList[1])

    for i in range(K):
        centroids[i] = getCentroid(grpList[i])

    print(f'c0: {centroids[0]}, c1: {centroids[1]}, c2: {centroids[2]}')
    
    # centroids[0] = c0
    # centroids[1] = c1

    if (oldGrpList is not None and isEqualGrpList(grpList, oldGrpList)):
        doStop = True
    else:
        oldGrpList = grpList


#print(oldGrpList)
