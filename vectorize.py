import numpy as np

centroids = np.array([[3,3], [6,2], [8,5]])

def f(i):
    return i * i + 3 * i - 2 if i > 0 else i * 5 + 8

fv = np.vectorize(f)  # or use a different name if you want to keep the original f

result_array = fv(centroids)  # if A is your Numpy array

# squarer = lambda t: t ** 2
# sv = np.vectorize(squarer)

# result_array = sv(centroids)
print(result_array)