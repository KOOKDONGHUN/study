import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential

from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')

from keras.datasets import mnist
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

# print(f"x_train[0] : {x_train[0]}")

# print(f"x_train.shape : {x_train.shape}")
# print(f"x_test.shape : {x_test.shape}")

# print(f"y_train.shape : {y_train.shape}")
# print(f"y_test.shape : {y_test.shape}")

'''plt.imshow(x_train[0],'Blues')
plt.show()'''

# 데이터 전처리 1.   원-핫-인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(f"y_train.shape : {y_train.shape}") # y_train.shape : (60000, 10)

# x_train = x_train / 255 # minmax와 같은 말 

# x_train = x_train.reshape(x_train[0],x_train[1],x_train[2],1).astype('float32')/255 #TypeError: only integer scalar arrays can be converted to a scalar index
# x_test = x_test.reshape(x_test[0],x_test[1],x_test[2],1).astype('float32')/255

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255


# 2. 모델구성
model = Sequential()
model.add(Conv2D(10,(3,3), input_shape=(28,28,1)))
model.add(Dropout(0.8))

model.add(Conv2D(10,(3,3)))
model.add(Dropout(0.8))

model.add(Conv2D(10,(3,3)))
model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.summary()

# model.save('./model/model_test01.h5') # 모델만 저장됨 

# 3. 컴파일(훈련준비),실행(훈련)

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=10,batch_size=150,callbacks=[],validation_split=0.1)

model.save('./model/model_test01.h5') # 가중치 까지 저장됨 

# 4. 평가, 예측
loss_accuracy = model.evaluate(x_test,y_test,batch_size=150)

print(f"loss : {loss_accuracy[0]}")
print(f"accuracy : {loss_accuracy[1]}")

''' 피팅후 저장한 모델에는 가중치가 같이 저장된다.
    loss : 0.31936209874830473
    accuracy : 0.911300003528595
'''

''' 피팅 전에 저장한 모델은? 
    keras86번에서 실행한 결과 -> RuntimeError: You must compile a model before training/testing. Use `model.compile(optimizer, loss)`.
    86번에서피팅을 안했음
    loss : 0.3159060607943684
    accuracy : 0.911899983882904
'''

''' 핏팅후 저장된 값
loss : 0.3071962803043425
accuracy : 0.9139999747276306'''