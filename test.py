import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math

# test data
A = np.array([[4, 1, 1, 4], [1, 4, 2, 0], [2, 1, 4, 5], [1, 4, 1, 0]])

# print(A)

U, s, VT = np.linalg.svd(A)

# 대각 행렬로 변환
S = np.zeros(A.shape)
for i in range(len(s)):
    S[i][i] = s[i]
# 테스트용
# print(np.round_(U,2))
# print(np.round_(s,2))
# print(np.round_(VT,2))

# 테스트용
# Sigma = zeros((A.shape[0], A.shape[1]))
# Sigma[:A.shape[1], :A.shape[1]] = diag(s)

# B = U.dot(Sigma.dot(VT))
# print(B)


# 평균제곱차이 유사도
# def sim_msd(data, name1, name2):
#     sum = 0
#     count = 0
#     for i in data[name1]:
#             if i in data[name2]:
#                 sum += pow(data[name1][i] - data[name2][i], 2)
#                 count += 1
#     return 1 / ( 1 + (sum/count))

def similar_cosine(data, name1, name2):
    sum_name1 = 0
    sum_name2 = 0
    sum_name1_name2 = 0
    for i in data[name1]:
        if i in data[name2]:
            sum_name1 += pow(data[name1][i], 2)
            sum_name2 += pow(data[name2][i], 2)
            sum_name1_name2 += data[name1][i]* data[name2][i]

    return sum_name1_name2 / (math.sqrt(sum_name1) * math.sqrt(sum_name2))



