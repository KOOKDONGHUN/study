from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2) # 가장 많이 쓰는 데이터 1000개중에 0.2를 테스트로 사용

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print(x_train[0]) # [1, 2, 2, 8, 43, 10, 447, ... , 30, 32, 132, 6, 109, 15, 17, 12]
print(y_train[0]) # 3

# print(x_train[0].shape) # 뉴스 기사 한줄 -> 리스트이다.
print(len(x_train[0])) # 87개의 단어로 구성된 기사한줄

category = np.max(y_train) +1 # 인덱스는 0부터 
print(category) # 46개의 카테고리 (기사의 종류 [경제, 스포츠, .. ])

y_bunpo = np.unique(y_train) # y_train에 있는 값들중에 중복을 제거한 번호?같은거
print(y_bunpo)

# 뉴스의 카테고리가 몇개씩 분포돼있는지 확인
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count() # groupby 사용법 익히기 
print(bbb)
print(bbb.shape)

# 특정 카테고리에는 기사의 개수가 상대적으로 적거나 상대적으로 너무 많음 ...

# pad_sequance
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

print(len(x_train[-1]))

# x_train = pad_sequences(x_train, maxlen=100, padding='pre') # maxlen 최대 개수, truncating 자른다? 앞에서 부터? maxlen 보다 크다면 잘림
x_train = pad_sequences(x_train, maxlen=1000, padding='pre') # maxlen 최대 개수, truncating 자른다? 앞에서 부터? maxlen 보다 크다면 잘림
# x_test = pad_sequences(x_test, maxlen=100, padding='pre') # maxlen 최대 개수, truncating 자른다? 앞에서 부터? maxlen 보다 크다면 잘림
x_test = pad_sequences(x_test, maxlen=1000, padding='pre') # maxlen 최대 개수, truncating 자른다? 앞에서 부터? maxlen 보다 크다면 잘림
# x_train = pad_sequences(x_train) # 상관없음 

print(len(x_train[0]))
print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("ss",y_test.shape)

print(x_train.shape) # (8982, 100)
print(x_test.shape) # (2246, 100)

# 2. model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM

model = Sequential()

# word_size -> 1000
model.add(Embedding(1000, 128, input_length=100)) # input_length == maxlen
model.add(LSTM(64)) 
model.add(Dense(46, activation='softmax'))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=64, epochs=10,validation_split=0.2)

acc = model.evaluate(x_test,y_test)[1]
print(f'acc : {acc}')

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c ='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c ='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()