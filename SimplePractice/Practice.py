import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math
from numpy.linalg import norm


# Training set
A = np.array([[4, 1, 1, 4], [1, 4, 2, 0], [2, 1, 4, 5], [1, 4, 1, 0]])

# Test set
# [1,4,1,0]


data = {"user" : ['U1', 'U2', 'U3', "U4"],
        "I1" : [4,1,2,1],
        "I2" : [1,4,1,4],
        "I3" : [1,2,4,1],
        "I4" : [4,0,5,2]}
df = pd.DataFrame(data)

print(df)

# svd 적용
U, s , V = np.linalg.svd(A, full_matrices=True)


# 대각행렬 만들기
# S = np.zeros(A.shape)
# for i in range(len(s)):
#     S[i][i] = s[i]
S = np.diag(s)

# 테스트
appA = np.dot(U, np.dot(S,V))

print(appA)

# A 평점
# U 사용자
# s 특이값
# V 영화
# X.shape 로 배열의 행렬 길이를 알 수 있음

