from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',
         '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', # '최고에요' -> '참 최고에요'
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
         '재미없어요', '너무 재미없다', '참 재밋네요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)


x = token.texts_to_sequences(docs)
print(x)

# x = np.array(x)
# print(x.shape)

# 원핫 인코딩?
from keras.preprocessing.sequence import pad_sequences 

pad_x = pad_sequences(x, padding='pre') 
print(pad_x)

print(pad_x.shape)

pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
print(pad_x.shape)

print(labels.shape)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM

model = Sequential()
model.add(LSTM(3, input_shape=(5,1)))
model.add(Dense(1,activation='sigmoid'))

model.summary()

# compile, fit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print(f'acc : {acc}')