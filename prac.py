import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math
from numpy.linalg import norm


# Training set
A = np.array([[4, 1, 1, 4], [1, 4, 2, 0], [2, 1, 4, 5]])

# Test set
# [1,4,1,0]


data = {"user" : ['U1', 'U2', 'U3'],
        "I1" : [4,1,2],
        "I2" : [1,4,1],
        "I3" : [1,2,3],
        "I4" : [4,0,5]}
df = pd.DataFrame(data)

print(df)

# svd 적용
U, s , V = np.linalg.svd(A, full_matrices=True)

# print(U)
# print(s)
# print(V)

# 대각행렬 만들기
# S = np.zeros(A.shape)
# for i in range(len(s)):
#     S[i][i] = s[i]
S = np.diag(s)
# S가 3 X 3 의 형태임

U = U[0:3,0:3]

V = V[0:3,0:4]

# 테스트
appA = np.dot(U, np.dot(S,V))

print(appA)



