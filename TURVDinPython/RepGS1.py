import numpy as np 
import pandas as pd
import math
from numpy.linalg import norm
import scipy.linalg as lin
import geometric

def RepGS(V, v, gamma):
    print("RegGS 시작")
    gamma = 1
    n = V.shape[0]
    d = V.shape[1]

    if(v.shape[1] == 0):
        y = np.zeros((d,0))

    nr_o = np.linalg.norm(v)
    nr = np.dot(np.spacing(1),nr_o)
    y=np.zeros((d,1))
    # print("nr_o")
    # print(nr_o)
    
    if(d==0):
        if(gamma):
            v = v / nr_o
            
            y = nr_o
        else:
            y = np.zeros((0,1))
    # print("y")
    # print(y.shape)
    print("V")
    print(V.shape)
    print("v")
    print(v.shape)
    y = np.dot(V.transpose(),v)
    v = v - np.dot(V,y)
    nr_n = np.linalg.norm(v)
    while nr_n < (nr_o * 0.5) and nr_n > nr :
        s = np.dot(V.transpose(),v)
        v = v - np.dot(V,s)
        y = y+s
        nr_o = nr_n
        nr_n = np.linalg.norm(v)

        ort = ort + 1

    if(nr_n <= nr):
        if(ort > 2):
            print('dependence!')
        if(gamma):
            if(d<n):
                v = RepGS(V,np.random.rand(n,1),1)
                y = np.concatenate((n,0))
            else:
                v = np.zeros((n,0))
        else:
            v = np.dot(0,v)
    else:
        if(gamma):
            y = np.concatenate((y,nr_n), axis = None)
            v = v / nr_n
    return v, y

def cgs(V):
    # """Classical Gram-Schmidt (CGS) algorithm"""
    m, n = V.shape
    R = np.zeros((n, n))
    Q = np.empty((m, n))
    R[0, 0] = np.linalg.norm(V[:, 0])
    Q[:, 0] = V[:, 0] / R[0, 0]
    for k in range(1, n):
        R[:k-1, k] = np.dot(Q[:m, :k-1].T, V[:m, k])
        z = V[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
        R[k, k] = np.linalg.norm(z) ** 2
        Q[:m, k] = z / R[k, k]
    return Q, R