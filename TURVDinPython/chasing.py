import numpy as np

def chasing(L, v, u):

    m = max(u.shape)

    Q, R = np.qr(u)
    Q = np.fliplr(np.transpose(Q))

    Z, R = np.qr(v) 
    Z = np.fliplr(np.transpose(Z))
    Lnew = np.dot(np.dot(np.transpose(Q), L), Z)

    Q2, R = np.qr(np.transpose(Lnew[1:m-1, 1:m-1]))
    Q3 = np.eye(m,m)
    Q3[1:m-1, 1:m-1] = Q2

    Z = np.dot(Z, Q3)
    Lnew = np.transpose(R)

    return Q, Lnew, Z

