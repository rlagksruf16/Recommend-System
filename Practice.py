import numpy as np


# a = [[1,2,3], [4,5,6], [7,8,9]]
A = [[1, 1, 3], [2, 0, 4], [-1, 6, -1]]
B = [2, 19, 8]
arrA = np.array(A)
arrB = np.array(B)

x = arrB/arrA
# arr = np.array(a)
print(x.size)

# b = np.random.normal(size=(arr.shape[0],1))

print(x)