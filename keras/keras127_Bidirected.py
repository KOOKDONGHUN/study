from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. data

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1500,skip_top=1) # skip_top 최상위 단어 2개는 삭제하고 가져옴

print(x_train.shape, x_test.shape) # (2500,) (2500,)
print(y_train.shape, y_test.shape) # (2500,) (2500,)

print(x_train[0]) # [1, 2, 2, 8, 43, 10, 447, ... , 30, 32, 132, 6, 109, 15, 17, 12]
print(y_train[0]) # 1

# print(x_train[0].shape) # 
print(len(x_train[0])) # 218개의 단어로 구성된 

category = np.max(y_train) +1 # 인덱스는 0부터 
print(category) # 2개의 카테고리 
y_bunpo = np.unique(y_train) # y_train에 있는 값들중에 중복을 제거한 번호?같은거
print(y_bunpo) # [0 1]

# 영화 평가 정보
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count() # groupby 사용법 익히기 
print('bbb : ',bbb) # 
print('bbb.shape',bbb.shape) # (2,)


# 특정 카테고리에는 기사의 개수가 상대적으로 적거나 상대적으로 너무 많음 ...

# pad_sequance
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# mx = 0
# for i in range(len(x_train)):
#     l = len(x_train[i])
#     if mx < l:
#         mx = l
# print(f'mx : {mx}')

x_train = pad_sequences(x_train, maxlen=1500, padding='pre') # maxlen 최대 개수, truncating 자른다? 앞에서 부터? maxlen 보다 크다면 잘림
x_test = pad_sequences(x_test, maxlen=1500, padding='pre') # maxlen 최대 개수, truncating 자른다? 앞에서 부터? maxlen 보다 크다면 잘림

print(len(x_train[0])) # 200
print(len(x_train[-1])) # 200

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print("ss",y_test.shape) # (25000,)

print(x_train.shape) # (25000, 200)
print(x_test.shape) # (25000, 200)

# 2. model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, BatchNormalization, Dropout, Conv1D, MaxPooling1D

model = Sequential()

# word_size -> 1000
model.add(Embedding(2000, 512, input_length=1500)) # input_length == maxlen
# model.add(Embedding(2000, 128)) # input_length == maxlen

model.add(Conv1D(512,1))
model.add(MaxPooling1D())
model.add(BatchNormalization())

model.add(Conv1D(256,1))
model.add(MaxPooling1D())
model.add(BatchNormalization())

model.add(Conv1D(128,1))
model.add(MaxPooling1D())
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# compile
from keras.optimizers import Adam
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=64, epochs=10,validation_split=0.3,verbose=2)

acc = model.evaluate(x_test,y_test)[1]
print(f'acc : {acc}')

# predict
# word_index = imdb.get_word_index()

# word_index = {k : (v+3) for k,v in word_index.items()}
# word_index['<PAD>'] = 0
# word_index['<START>'] = 1
# word_index['<UNK>'] = 2
# word_index['<UNUSED>'] = 3

# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# def decode_review(text):
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])

# plot
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c ='red', label='ValSet Loss')
plt.plot(y_loss, marker='.', c ='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

''' 첫번째 해야 할것imdb데이터 내용을 확인, 데이터 구조 파악학기 
y값과 x값이 몇바이 몇인지도 확인하기.
#word_size 전체 데이터 부분  변경해서 최상값 확인. 
주관과제 groupby()의 사용법 숙지할것  '''