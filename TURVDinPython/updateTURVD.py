import numpy as np
from numpy.linalg import norm
import math
import scipy.linalg as lin

import RepGS1
import chasing

def updateTURVD(X, U, R, V, x, normE, ep): 
    m, n = X.shape
    u, h = RepGS1.RepGS(U,x,1)

    Us = np.concatenate((U,u))
    Rs = np.concatenate((np.concatenate((R,np.zeros((1,R.shape[1]))),h)))
    Vs = np.vstack((np.hstack((V,np.zeros((V.shape[0],1)))),np.hstack((np.zeros((1,V.shape[1])),1))))

    # Rs = [[R; zeros(1, size(R,2))], h];
    # Vs = [V, zeros(size(V,1),1); zeros(1,size(V,2)), 1];

    sigk1, u1, v1 = power_iter(np.linalg.inv(Rs))

    if(abs(sigk1) > 0 ):
        sigk1 = 1 / sigk1

    if(np.sqrt(sigk1**2 + normE**2) > ep):
        normEs = normE
    else:
        P2, Rs, Q3 = chasing.chasing(Rs, v1, u1)
        Us = np.dot(Us, P2[:, 1:-2])
        Vs = np.dot(Vs, Q3[:, 1:-2])
        normEs = np.sqrt(sigk1**2 + normE**2)
    return Us, Rs, Vs, normEs




def power_iter(X):
    citer = 50


    v = np.random.normal(size=(X.shape[1],1))
    u = np.random.normal(size=(X.shape[0],1))
    sigk = 0
    eps = 1.0e-4
    
    for _ in range(1,citer+1):
        v2 = np.dot(X,u)
        sigma = np.linalg.norm(v2)
        v2 = v2 / sigma
        Xx = np.transpose(X)
        u2 = np.dot(Xx,v2)
        u2 = u2 / np.linalg.norm(u2)
        u = u2
        v = v2
        if(abs(sigk - sigma) < eps ):
            break
        
    return sigma, u, v
        
        
