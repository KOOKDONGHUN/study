import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import load_model

# 데이터 전처리 1.   원-핫-인코딩
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

# 2. 모델구성
model = load_model('./model/model_test01.h5')
model.summary()

'''
model = Sequential()
model.add(Conv2D(10,(3,3), input_shape=(28,28,1)))
model.add(Dropout(0.8))

model.add(Conv2D(10,(3,3)))
model.add(Dropout(0.8))

model.add(Conv2D(10,(3,3)))
model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.summary()'''


# 3. 컴파일(훈련준비),실행(훈련)
'''
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs=10,batch_size=150,callbacks=[],validation_split=0.1)'''

'''컴파일과 핏을 하기전에 모델을 저장한 경우 컴파일과 핏을 해줘야함 결과값은 동일하지 않을 수 있다.'''

# 4. 평가, 예측
loss_accuracy = model.evaluate(x_test,y_test,batch_size=150)

print(f"loss : {loss_accuracy[0]}")
print(f"accuracy : {loss_accuracy[1]}")

''' 핏팅후 저장된 모델의 값
loss : 0.3071962803043425
accuracy : 0.9139999747276306'''

''' 불러온 모델을 실행한 결과값이 동일함을 알 수 있다 때문에 모델만 저장된것이 아니라가중치도 같이 저장되었다고 볼수 있음 
loss : 0.3071962803043425
accuracy : 0.9139999747276306'''

