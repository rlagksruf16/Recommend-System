import numpy as np 
import pandas as pd
import math
from numpy.linalg import norm
import scipy.linalg as lin
import geometric

def RepGS(V, v, gamma):
    gamma = 1
    n = V.shape[0]
    d = V.shape[1]

    if(v.shape[1] == 0):
        y = np.zeros((d,0))

    nr_o = np.linalg.norm(v)
    nr = np.dot(np.spacing(1),nr_o)
    y=np.zeros((d,1))

    if(d==0):
        if(gamma):
            v = v / nr_o
            y = nr_o
        else:
            y = np.zeros((0,1))

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
                v = RepGS(V,np.random.rand(n,1),)
                y = np.concatenate((n,0))
            else:
                v = np.zeros((n,0))
        else:
            v = np.dot(0,v)
    else:
        if(gamma):
            y = np.concatenate((y,nr_n), axis = None)
            v = v / nr_n