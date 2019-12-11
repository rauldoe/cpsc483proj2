# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:45:00 2019

@author: shlakhanpal
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,1],[2,1],[4,3],[5,4]]) # class activity

print('Printing data in X...')
print(X)

print('Dimensions of X', X.shape)

num_points = len(X)
print("Total number of points in dataset, ie. X:", num_points)

K = 2 # class activity

initial_centroids = np.array([[1,1],[2,1]]) # class activity

plt.plot(X[:, 0], X[:, 1], 'go')
plt.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'rx')

print("X first row ", X[0])

dist = np.zeros((K,num_points))

def distanceBetPointsAndCentroids():
    for i in range(K):
        for j in range(num_points):
            dist[i,j] = np.linalg.norm(initial_centroids[i] - X[j])
            print (dist[i,j])
            
distanceBetPointsAndCentroids()        




