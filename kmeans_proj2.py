import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# def f(i):
#     return i * i + 3 * i - 2 if i > 0 else i * 5 + 8

# fv = np.vectorize(f)  # or use a different name if you want to keep the original f

# result_array = fv(centroids)  # if A is your Numpy array

# squarer = lambda t: t ** 2
# sv = np.vectorize(squarer)

# result_array = sv(centroids)
# print(result_array)

def distanceBetweenPointsAndCentroids(X, centroids, K, dist):

    num_points = len(X)
    # print(f'Total number of points in the dataset, ie. X is {num_points}')

    for indexOfCentroids in range(K):
        for indexOfPoints in range(num_points):
            dist[indexOfCentroids, indexOfPoints] = np.linalg.norm(centroids[indexOfCentroids] - X[indexOfPoints])
    return dist

def selectCentroidMemberships(X, dist, K):
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

def recalculateCentroid(centroidMembership):
    totalX = 0.0
    totalY = 0.0
    for i in range(len(centroidMembership)):
        totalX += centroidMembership[i][0]
        totalY += centroidMembership[i][1]

    return [totalX/len(centroidMembership), totalY/len(centroidMembership)]

def doPlot(plt, X, centroids):
    plt.plot(X[:, 0], X[:, 1], 'go')
    plt.plot(centroids[:, 0], centroids[:, 1], 'rx')
    plt.show()

def doPlotWithOriginal(plt, X, originalCentroids, centroids):
    plt.plot(X[:, 0], X[:, 1], 'go')
    plt.plot(originalCentroids[:, 0], originalCentroids[:, 1], 'rx')
    plt.plot(centroids[:, 0], centroids[:, 1], c='b', marker='x', linestyle='None')
    plt.show()

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

originalCentroids = centroids.copy()
currentCentroids = centroids.copy()

doPlot(plt, X, centroids)

while doStop == False:
    dist = distanceBetweenPointsAndCentroids(X, centroids, K, dist)

    centroidMemberships = selectCentroidMemberships(X, dist, K)

    for i in range(K):
        centroids[i] = recalculateCentroid(centroidMemberships[i])

    print(f'c0: {centroids[0]}, c1: {centroids[1]}, c2: {centroids[2]}')
    
    doPlotWithOriginal(plt, X, originalCentroids, centroids)

    if (np.array_equal(centroids, currentCentroids)):
        doStop = True
    else:
        currentCentroids = centroids.copy()

#print(oldGrpList)
