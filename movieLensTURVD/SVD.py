# demo for updating singular value decomposition
import numpy as np 
import pandas as pd
import math
from numpy.linalg import norm
import scipy.linalg as lin
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import updateTURVD.py

data = pd.io.parsers.read_csv('/Users/hankyul/developer/Recommend-System/ml-1m/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('/Users/hankyul/developer/Recommend-System/ml-1m/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')

# 파싱된 데이터 매트릭스로 변형
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

# 배열 확인
print(ratings_mat.shape)
# print(len(ratings_mat))
# print(len(ratings_mat[0]))


Ri = ratings_mat[:,len(ratings_mat[0])-100:]
# 100개 뽑은 것
Rt = ratings_mat[:,0:len(ratings_mat[0])-100]

# print(Ri.shape)
# print(Rt.shape)

U, S, V = np.linalg.svd(Ri, full_matrices=False)

# print(U.shape)
# print(S.shape)
# print(V.shape)

epsilon = 0.1

if(epsilon == 0):
    epsilon = math.sqrt(sum(np.diag(S) ** 2)) * 0.1

sum = 0

max_sum = np.linalg.norm(np.diag(S), axis=None, keepdims=False) ** 2

S = np.diag(S)

for idx in range(1, S.shape[0]+1):
    sum = sum + S[idx-1][idx-1] ** 2
    if(math.sqrt(max_sum - sum) < epsilon * 0.9):
        break

Ui = U[:, 1:idx+1]
Si = S[1:idx+1, 1:idx+1]
Vi = V[:,1:idx+1]

normE = np.linalg.norm(Ri - (np.dot(np.dot(Ui,Si),np.transpose(Vi))), axis=None, keepdims=False)

for i in range(1,Rt.shape[1]):
    x = Rt[:i+1]
    Ui, Si, Vi, normE = updateTURVD(Ri, Ui, Si, Vi, x, normE, epsilon)
    Ri = np.concatenate((Ri, x))
    print('%d th iteration, err=%f, trunc=%d'%(i,np.linalg.norm(Ri - (np.dot(np.dot(Ui,Si),np.transpose(Vi))), axis=None, keepdims=False),Ui.shape[1]) )
    