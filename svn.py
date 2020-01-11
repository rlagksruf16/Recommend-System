import numpy as np 
import pandas as pd

# A의 사이즈
# 시간 측정 하는걸
# tictok
# svd 양쪽 사이드
# svd 시작하기 전
# svd 시작하고 난 뒤 시간측정 
# false 해보고
# 실제 데이터셋 사이즈 몇인지 
# svd 말고 pca 로 한번 돌려보기
# 소스 코드 
# svd 시간 
# A행렬 사이즈 가로 세로 어떤 데이터 가리키는지 실제 데이터에서 -> 문서화 해서 보내주시고
# 1000의 데이터를 svd faslse 로 하시면 
# 이것저것 테스트 svd full 을 false 로 했을때 돌아가는지 
# 1000개를 샘플링해서 돌리는데 
# 나중에는 실험할때 랜덤하게 뽑아야되니까 돌리는게 우선이니까 10만개가 안돌아가면 1000개만 돌려보기 
# 실제로 svd 돌아가는지 여부 + pca 여부
# pca 랑 svd 똑같은 데이터셋 넣었을 때 차이가 얼마나 발생하는지
# 데이터셋 -> 다운받았는지 어디서 pca 

# svd 행렬 두번 곱해서 돌리면 pca 가 된다 (물론 평균을 빼야되지만)
# 돌아가는지 부터 확인해서 문서로 해서 보내기 -> 다음주까지 


# 파일 받아서 읽기 -> 파싱
data = pd.io.parsers.read_csv('C:/ml-1m/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('C:/ml-1m/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')

data.head()

# 매트릭스로 해결!
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

# 정규화
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

# svd 적용
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

# -------------------------유사도 검사 부분--------------------
# 유사도 검사!
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# 코사인 유사도 활용

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])

k = 30
movie_id = 1 # 유사한 결과를 찾고 싶은 품목
top_n = 10   # 위에서부터 몇개 출력할지 결정

sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, indexes)