import scipy.linalg as lin
import numpy as np
import math
import sys

np.set_printoptions(threshold=sys.maxsize)

def geometry(m,n,kappa):
    p = min([m,n])
    Inkappa = -math.log(kappa)/(p-1)
    exponents = list(range(0,p))
    sigma = [math.exp(i * Inkappa) for i in exponents]

    if m > n:
        U = lin.orth(np.random.rand(m,n))
        V = lin.orth(np.random.rand(n,n))
    else:
        U = lin.orth(np.random.rand(m,m))
        V = lin.orth(np.random.rand(n,m))
    # sys.stdout = open('output1.txt', 'w')
    X = np.dot(np.dot(U, np.diag(sigma)), V.transpose())
    print(X)
    return X



