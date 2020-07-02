# 20-07-02_27
# 한 개의 모델에서 분류와 회귀를 동시에 나오게 할 수 있을까?


### 1. 데이터
import numpy as np

x_train = np.arange(1,1001,1)
y_train = np.array([0,1]*500)


from keras.utils.np_utils import to_categorical
### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

y_train =to_categorical(y_train)

model = Sequential()

model.add(Dense(512,input_shape=(1,)))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(2,activation="sigmoid"))


### 3. 실행, 훈련
model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=50, batch_size=16,validation_split=0.1)


### 4. 평가, 예측
loss = model.evaluate(x_train, y_train )
print('loss :', loss)

x_pred = np.array([11, 12, 13, 14])

y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)