import numpy as np 
import pandas as pd
from numpy import zeros
from numpy import diag
from numpy import array
from numpy import dot
import math
from numpy.linalg import norm
import timeit


data = pd.io.parsers.read_csv('C:/ml-1m/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('C:/ml-1m/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')


# 매트릭스로 해결!
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

A = ratings_mat[0:1000,0:1000]

print(A.shape)

U, s , V = np.linalg.svd(A, full_matrices=True)


S = np.diag(s)


U = U[:,0:5]
S = S[0:5,0:5]
V = V[0:5,:]

appA = np.dot(U, np.dot(S,V))

def cosine_similarity(data,x):
    z = 0
    for i in range(0,len(data)):
        if(x==i):
            continue
        else:
            if(z < dot(appA[x],appA[i]/(norm(appA[x]) * norm(appA[i])))):
                z = dot(appA[x],appA[i]/(norm(appA[x]) * norm(appA[i])))
    return z

similar = cosine_similarity(appA,0)

print(similar)
