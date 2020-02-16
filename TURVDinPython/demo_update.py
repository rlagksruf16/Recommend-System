# demo for updating singular value decomposition
import numpy as np 
import pandas as pd
import math
from numpy.linalg import norm
import scipy.linalg as lin
import geometric


m = 500
n = 50
window = 20
kappa = 1.0e+5
epsilon = 0.1

# 벡터 랜덤으로 생성해서 txt 파일에 저장하고 그것을 매트랩 코드랑 비교해서 결과값이 비슷하면 됨
X = geometric.geometry(m,n,kappa)

# Test Code
# print(X)
# print(type(X))

Xi = X[:, 1:n-window]
xdata = X[:,(n-window+1):]

U, R, V = np.linalg.svd(Xi, full_matrices=True)

if(epsilon == 0):
    epsilon = math.sqrt(sum(np.diag(R) ** 2)) * 0.1
#   truncation..0
#  [idx, ~] = find(diag(R) < epsilon / 5);
sum = 0
max_sum = np.linalg.norm(R, ord = 'fro', axis=None, keepdims=False) ** 2
for idx in range(1,np.size(R,1)+1):
    sum = sum + R[idx-1,idx-1] ** 2
    if(math.sqrt(max_sum - sum) < epsilon * 0.9):
        break
Us = U[:, 1:idx]
Rs = R[1:idx, 1:idx]
Vs = V[:, 1:idx]
normE = np.linalg.norm(Xi - (np.dot(Us, np.dot(Rs,Vs))), ord='fro', axis=None, keepdims=False)

