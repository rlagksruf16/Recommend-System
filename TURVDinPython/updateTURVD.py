# power Iteration matlab 코드

import numpy as np
from numpy.linalg import norm

def power_iter(X):
    citer = 50
# 원래 무한대로 곱해야 되는데 50번만 곱해도 유의미한 결과값이 나오기 때문에 50번만 곱한다.

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
        
        




# function [sigma, u, v] = power_iter(X)
#     citer = 50; 
#     //무한대로 곱해야하는데 citer 50번만 곱한다
#     v = randn(size(X,2),1); 
#     u = randn(size(X,1),1);
#     sigk = 0;
#     eps = 1.0e-4;
#     for i = 1:citer
#         v2 = X * u;
#         sigma = norm(v2); v2 = v2 / sigma;
#         u2 = X' * v2;
#         X' 의 '는 켤레 전치를 의미한다.
#         u2 = u2 / norm(u2);
#         u = u2;
#         v = v2;
#         if abs(sigk - sigma) < eps 
#             break;
#         end
#     end