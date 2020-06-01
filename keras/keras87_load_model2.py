'''keras86 copy'''
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model

# 데이터 전처리 1.   원-핫-인코딩
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

# 2. 모델 구성

model = load_model('./model/model_test01.h5')
model.add(Dense(10,name='1'))
model.add(Dense(10,name='2'))
model.add(Dense(10,activation='softmax',name='output'))

model.summary() #

# 3. 컴파일, 실행 
'''model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(x_train,y_train,epochs=10,batch_size=150,callbacks=[],validation_split=0.1)
레이어가 적용되는건지 모르겠네 ... 적용된다고 말씀하심 핏과 컴파일은 다시 할 필요가 없음 '''

# 4. 평가, 예측
loss_accuracy = model.evaluate(x_test,y_test,batch_size=150)

print(f"loss : {loss_accuracy[0]}")
print(f"accuracy : {loss_accuracy[1]}")

''' 핏팅후 저장된 값
loss : 0.3071962803043425
accuracy : 0.9139999747276306'''

''' 현재 파일의 실행결과 -> 추가된 레이어 적용된것을 알 수 있음 
loss : 2.2991785430908203
accuracy : 0.027400000020861626'''