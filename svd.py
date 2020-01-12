import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math
from numpy.linalg import norm

B = np.array([[4,1,1,4], [1,4,2,0] ,[2,1,4,5] ,[1,4,1,2]])

U, s , V = np.linalg.svd(B, full_matrices=True)

# print(U)
# print(s)
# print(V)

S = np.diag(s)
# S가 3 X 3 의 형태임

U = U[:,0:3]
S = S[0:3,0:3]
V = V[0:3,:]


appA = np.dot(U, np.dot(S,V))

print(appA)

def cosine_similarity(data,x):
    z = 0
    for i in range(0,len(data)):
        if(x==i):
            continue
        else:
            if(z < dot(appA[x],appA[i]/(norm(appA[x]) * norm(appA[i])))):
                z = dot(appA[x],appA[i]/(norm(appA[x]) * norm(appA[i])))
    return z

similar = cosine_similarity(appA,3)

print(similar)


# cos_sim = dot(appA[2],appA[3]/(norm(appA[2]) * norm(appA[3])))

# print(cos_sim)
# 코사인 유사도 검사 함수
# def similar_cosine(data, name1, name2):
#     sum_name1 = 0
#     sum_name2 = 0
#     sum_name1_name2 = 0
#     for i in len(data):
#         if i in len(data):
#             sum_name1 += pow(data[name1][i], 2)
#             sum_name2 += pow(data[name2][i], 2)
#             sum_name1_name2 += data[name1][i]* data[name2][i]

#     return sum_name1_name2 / (math.sqrt(sum_name1) * math.sqrt(sum_name2))

# X = similar_cosine(appA,1,3)    
# print(X)

