import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math
from numpy.linalg import norm
import timeit

B = np.array([[4,1,1,4], [1,4,2,0] ,[2,1,4,5] ,[1,4,1,2]])

#svd 실행
start = timeit.default_timer()  #시작 시간
U, s , V = np.linalg.svd(B, full_matrices=True)
stop = timeit.default_timer()   #끝나는 시간


print(stop-start)
# print(U)
# print(s)
# print(V)

S = np.diag(s)
# S가 3 X 3 의 형태임


# svd 가 제대로 되엇는지 확인용
# appA1 = np.dot(U, np.dot(S,V))

# print(appA1)

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


