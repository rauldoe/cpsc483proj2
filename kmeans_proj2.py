import numpy as np
import matplotlib.pyplot as plt
import scipy.io

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

def plotCentroids(plt, centroids, colors):
    
    for i in range(len(centroids)):
        plt.scatter(centroids[i, 0], centroids[i, 1], c=colors[i], marker='x', linestyle='None', s=100)
    
    return plt

def doPlotCentroids(plt, centroids, colors):
    plt = plotCentroids(plt, centroids, colors)
    plt.xlim([-1, 10])
    plt.ylim([-1, 10])
    plt.show()

def doInitialPlot(plt, centroids, colors, X):
    
    plt.plot(X[:, 0], X[:, 1], 'go')
    plt = plotCentroids(plt, centroids, colors)
    plt.xlim([-1, 10])
    plt.ylim([-1, 10])
    plt.show()

def doPlotMatchingXWithCentroid(plt, centroids, colors, centroidMemberships):
    
    for i in range(len(centroids)):
        plt.scatter(centroids[i, 0], centroids[i, 1], c=colors[i], marker='x', linestyle='None', s=100)
        membership = np.array(centroidMemberships[i])
        plt.plot(membership[:,0], membership[:,1], c=colors[i], marker='o', linestyle='None')
    plt.xlim([-1, 10])
    plt.ylim([-1, 10])
    plt.show()

datafile = 'kmeansdata.mat'
points = scipy.io.loadmat(datafile)

X = points['X']

K = 3

centroids = np.array([[3,3], [6,2], [8,5]])

numberOfPoints = len(X)
dist = np.zeros((K, numberOfPoints))

doStop = False

originalCentroids = centroids.copy()
currentCentroids = centroids.copy()
colors=["#0000FF", "#00FF00", "#FF0066"]

doInitialPlot(plt, centroids, colors, X)

doOnce = False

while doStop == False:
    dist = distanceBetweenPointsAndCentroids(X, centroids, K, dist)

    centroidMemberships = selectCentroidMemberships(X, dist, K)

    for i in range(K):
        centroids[i] = recalculateCentroid(centroidMemberships[i])

    print(f'c0: {centroids[0]}, c1: {centroids[1]}, c2: {centroids[2]}')
    
    if (not doOnce):
        doPlotMatchingXWithCentroid(plt, centroids, colors, centroidMemberships)
        doOnce = True

    doPlotCentroids(plt, centroids, colors)

    if (np.array_equal(centroids, currentCentroids)):
        doStop = True
    else:
        currentCentroids = centroids.copy()

doPlotMatchingXWithCentroid(plt, currentCentroids, colors, centroidMemberships)
