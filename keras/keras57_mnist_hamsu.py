import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from keras.datasets import mnist


data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

# 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

# 2. 모델구성
input1 = Input(shape=(28,28,1))
fl1 = (Flatten())(input1)
dense1 = Dense(560,activation='relu')(fl1)
dense1 = Dropout(0.2)(dense1)

# dense1 = Dense(100)(dense1)
# dense1 = Dense(100)(dense1)

output1 = Dense(10,activation='softmax')(dense1)
model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)

els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=12,batch_size=110,callbacks=[],verbose=2)

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