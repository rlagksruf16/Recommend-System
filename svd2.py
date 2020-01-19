import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math
from numpy.linalg import norm
import timeit

# 데이터 파싱
data = pd.io.parsers.read_csv('C:/ml-1m/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('C:/ml-1m/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')


# 파싱된 데이터 매트릭스로 변형
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

# 확인 작업
# print(ratings_mat.shape)


# SVD
start = timeit.default_timer()
U, s , V = np.linalg.svd(ratings_mat, full_matrices=True)
stop = timeit.default_timer() 

# s의 형태 확인
# print(s.shape)

# s 대각행렬로 변환
S = np.diag(s)

# s의 모형에 따른 차원축소
U = U[:,0:3952]
S = S[0:3952,0:3952]
V = V[0:3952,:]

# 하나의 매트릭스로 묶기
appA = np.dot(U, np.dot(S,V))


# 코사인 유사도 함수
def cosine_similarity(data,x):
    z = 0
    sim_val = 0
    for i in range(0,len(data)):
        if(x==i):
            continue
        else:
            normA = norm(appA[x])
            if(normA != 0):
                if(z < dot(appA[x],appA[i]/(norm(appA[x]) * norm(appA[i])))):
                    z = dot(appA[x],appA[i]/(norm(appA[x]) * norm(appA[i])))
                    sim_val = i
    return z, sim_val

# 실행
similar = cosine_similarity(appA,1)

print(similar)
