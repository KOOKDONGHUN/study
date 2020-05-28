import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from keras.datasets import mnist


# 1. 데이터
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_pixels = x_train.shape[1] * x_train.shape[2]

# 각 픽셀에 대한 값을 0~255의 숫자를 0과1사이의 값으로 만들어줌
''' 여러가지 해보기 (784,1), (28,28), (392,2), (196,4) '''
# x_train = x_train.reshape(60000,num_pixels,1).astype('float32')/255
x_train = x_train.reshape(60000,28,28).astype('float32')/255
# x_test = x_test.reshape(10000,num_pixels,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28).astype('float32')/255


# 2. 모델구성
model = Sequential()
# model.add(LSTM(5,input_shape=(num_pixels,1)))
model.add(LSTM(5,input_shape=(28,28)))
model.add(Dense(512))
model.add(Dense(10,activation='softmax'))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)

els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=10,batch_size=200,callbacks=[],verbose=2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('keras54 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
# plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=30)

print(f"loss : {loss}")
print(f"acc : {acc}")