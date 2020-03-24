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
print(len(X))
print(len(X[0]))
print(X.shape)
Xi = X[:, 1:n-window]
print(Xi.shape)

xdata = X[:,(n-window+1):]

U, R, V = np.linalg.svd(Xi, full_matrices=False)

if(epsilon == 0):
    epsilon = math.sqrt(sum(np.diag(R) ** 2)) * 0.1
#   truncation..0
#  [idx, ~] = find(diag(R) < epsilon / 5);
sum = 0
# max_sum = np.linalg.norm(R, ord = 'fro', axis=None, keepdims=False) ** 2
max_sum = np.linalg.norm(np.diag(R), axis=None, keepdims=False) ** 2

R = np.diag(R)
# for idx in range(1,np.size(R,1)+1):
for idx in range(1, R.shape[0]+1):
    # idx = 1부터 R의 행 부분의 길이까지
    sum = sum + R[idx-1][idx-1] ** 2
    if(math.sqrt(max_sum - sum) < epsilon * 0.9):
        break

Us = U[:, 1:idx]
Rs = R[1:idx, 1:idx]
Vs = V[:,1:idx]

normE = np.linalg.norm(Xi - (np.dot(np.dot(Us,Rs),np.transpose(Vs))), axis=None, keepdims=False)

# for _ in range(1,):


# for i = 1:size(xdata,2)
#    x = xdata(:,i); 
#    [Us, Rs, Vs, normE] = updateTURVD(Xi,Us, Rs, Vs, x, normE, epsilon);
#    Xi = [Xi, x];
#    fprintf('%d th iteration, err = %g, trunc = %d\n',i, norm(Xi - Us * Rs * Vs','fro'), size(Us,2));
# end