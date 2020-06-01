'''keras91 copy'''
''' x와 y를 저장하기 '''
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential,load_model
from keras.datasets import mnist
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import np_utils

# 데이터 전처리 1. 불러와서 나누기 ( 전처리후 저장하였음 -> 데이터의 크기 때문에? 전처리 전에 데이터를 저장하는게 좋다라고 선생님이 말씀하심 )
x_train = np.load('./data/mnist_train_x.npy')
y_train = np.load('./data/mnist_train_y.npy')

x_test = np.load('./data/mnist_test_x.npy')
y_test = np.load('./data/mnist_test_y.npy')

print(f"x_train : \n{x_train}")
print(f"x_train.shape : {x_train.shape}")
print("x_test : \n",x_test)
print("x_test.shape : ",x_test.shape)

print("y_train :\n",y_train)
print("y_train.shape :",y_train.shape)
print("y_test : \n",y_test)
print("y_test.shape : ",y_test.shape)

# 데이터 전처리 2. 원-핫-인코딩
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# 데이터 전처리 3. 정규화
# x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
# x_test = x_test.reshape(10000,28,28,1).astype('float32')/255



# 2. 모델구성
model = load_model('./model/08--0.2648.hdf5')
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

modelpath = './model/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

hist = model.fit(x_train,y_train,epochs=10,batch_size=150,callbacks=[chpoint],validation_split=0.1)

model.save('./model/model_test01.h5') # 가중치 까지 저장됨 
'''

# 4. 평가, 예측
loss_accuracy = model.evaluate(x_test,y_test,batch_size=150)

print(f"loss : {loss_accuracy[0]}")
print(f"accuracy : {loss_accuracy[1]}")

''' 저장된 가중치
loss : 0.3134315144922584
accuracy : 0.9146000146865845'''

''' 불러온 후 실행 결과 -> 다르다
loss : 0.31044199684634805
accuracy : 0.9150000214576721'''