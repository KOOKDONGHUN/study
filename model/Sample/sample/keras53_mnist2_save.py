import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 전처리 1.   원-핫-인코딩
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

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

# 3. 컴파일(훈련준비),실행(훈련)
els = EarlyStopping(monitor='loss', patience=5, mode='auto')

modelpath = './model/Sample/model/MNIST/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs=30,batch_size=150,callbacks=[chpoint,els],validation_split=0.1)

model.save('./model/Sample/model/MNIST/keras53_model.h5') # 가중치 까지 저장됨
model.save_weights('./model/Sample/model/MNIST/keras53_weight.h5') # 가중치만 저장됨
modelpath = './model/Sample/model/MNIST/{epoch:02d}--{val_loss:.4f}.hdf5'

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=150)

print(f"loss : {loss}")
print(f"acc : {acc}")

''' 99.25이상 만들기 '''