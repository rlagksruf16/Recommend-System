import numpy as np
from numpy.linalg import norm

# a = [[1,2,3], [4,5,6], [7,8,9]]
# A = [[1, 1, 3], [2, 0, 4], [-1, 6, -1], [1,2,3]]
# B = [2, 19, 8]
# arrA = np.array(A)
# arrB = np.array(B)

# x = arrB/arrA
# arr = np.array(a)


# b = np.random.normal(size=(arr.shape[0],1))

# print(max(arrA.shape))


A = [[1,2], [3,4], [5,6], [7,8]]
arrA = np.array(A)
print(arrA.shape)

print(" ")

U, R, V = np.linalg.svd(arrA, full_matrices=False)
R = np.diag(R)
print(U)
print(R)
print(V)