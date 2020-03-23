import numpy as np 
import pandas as pd
import math
from numpy.linalg import norm
import scipy.linalg as lin
import geometric

def RepGS(V, v, gamma):

    # if(nargin < 3):
    #     gamma = 1


    n = V.shape[0]
    d = V.shape[1]

    if(v.shape[1] == 0):
        y = np.zeros((d,0))

    nr_o = np.linalg.norm(v)
    nr = np.dot(np.spacing(1),nr_o)
    y=np.zeros(0,1)

    if(d==0):
        if(gamma):
            v = v / nr_o
            y = nr_o
        else:
            y = np.zeros((0,1))
    
    y = np.dot(V.transpose(),v)
    v = v - np.dot(V,y)
    nr_n = np.linalg.norm(v)
    ort=0

    while nr_n < (nr_o * 0.5) and nr_n > nr :
        s = np.dot(V.transpose(),v)
        v = v - np.sdot(V,s)
        y = y+s
        nr_o = nr_n
        nr_n = np.linalg.norm(v)

        ort = ort+1
    
    if(nr_n <= nr):
        if(ort > 2):
            print('dependence!')
        if(gamma):
            if(d<n):
                v = RepGS(V,np.random.rand(n,1),)
                y = np.concatenate((n,0))
            else:
                v = np.zeros((n,0))
        else:
            v = np.dot(0,v)
    else: 
        if(gamma):
            # v = 
            y = np.concatenate((y,nr_n))
    return v,y




# import numpy as np
# from scipy import linalg as lin
# import math
# from numpy.linalg import norm


# def cgs(A):
#     # """Classical Gram-Schmidt (CGS) algorithm"""
#     m, n = A.shape
#     R = np.zeros((n, n))
#     Q = np.empty((m, n))
#     R[0, 0] = linalg.norm(A[:, 0])
#     Q[:, 0] = A[:, 0] / R[0, 0]
#     for k in range(1, n):
#         R[:k-1, k] = np.dot(Q[:m, :k-1].T, A[:m, k])
#         z = A[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
#         R[k, k] = linalg.norm(z) ** 2
#         Q[:m, k] = z / R[k, k]
#     return Q, R

# if __name__ == '__main__':

#     n = 5
#     X = np.random.random((n, n))
#     import rogues
# #    X = rogues.hilb(n)
#     Q, R = cgs(X)
#     assert np.allclose(np.dot(Q, R), X)
#     print np.linalg.norm(np.dot(Q.T, Q) - np.eye(5), np.inf)

# def gram_schmidt(X):
#     O = np.zeros(X.shape)
#     for i in range(X.shape[1]):
#         # orthogonalization
#         vector = X[:, i]
#         space = O[:, :i]
#         projection = vector @ space
#         vector = vector - np.sum(projection * space, axis=1)
#         # normalization
#         norm = np.sqrt(vector @ vector)
#         vector /= abs(norm) < 1e-8 and 1 or norm
        
#         O[:, i] = vector
#     return O

# X = [[1,1,1],[2,0,0],[0,0,1]]
# arrA = np.array(X)

# print(gram_schmidt(arrA))



# def cgs(A):
#     # """Classical Gram-Schmidt (CGS) algorithm"""
#     m, n = A.shape
#     R = np.zeros((n, n))
#     Q = np.empty((m, n))
#     R[0, 0] = np.linalg.norm(A[:, 0])
#     Q[:, 0] = A[:, 0] / R[0, 0]
#     for k in range(1, n):
#         R[:k-1, k] = np.dot(Q[:m, :k-1].T, A[:m, k])
#         z = A[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
#         R[k, k] = np.linalg.norm(z) ** 2
#         Q[:m, k] = z / R[k, k]
#     return Q, R


# abs = 5
# X = np.random.rand(abs, abs)
# b = np.array(X)
# Q, R = cgs(X)
# assert np.allclose(np.dot(Q, R), X)
# print (np.linalg.norm(np.dot(Q.T, Q) - np.eye(5), np.inf))


