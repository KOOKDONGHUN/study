from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. data

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1500,skip_top=1) # skip_top 최상위 단어 2개는 삭제하고 가져옴

# print(x_train.shape, x_test.shape) # (2500,) (2500,)
# print(y_train.shape, y_test.shape) # (2500,) (2500,)

# print(x_train[0]) # [1, 2, 2, 8, 43, 10, 447, ... , 30, 32, 132, 6, 109, 15, 17, 12]
# print(y_train[0]) # 1

# print(x_train[0].shape) # 
# print(len(x_train[0])) # 218개의 단어로 구성된 

category = np.max(y_train) +1 # 인덱스는 0부터 
# print(category) # 2개의 카테고리 
y_bunpo = np.unique(y_train) # y_train에 있는 값들중에 중복을 제거한 번호?같은거
# print(y_bunpo) # [0 1]

# 영화 평가 정보
y_train_pd = pd.DataFrame(y_train)
print(y_train_pd)
bbb = y_train_pd.groupby(0)[0].count() # groupby 사용법 익히기 
print('bbb : ',bbb) # 
print('bbb.shape',bbb.shape) # (2,)