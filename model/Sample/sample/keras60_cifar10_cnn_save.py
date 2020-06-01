from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D , Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255

# 2. 모델구성
input1 = Input(shape=(32,32,3))

dense1 = (Conv2D(32,(3,3),activation='relu'))(input1)
dense1 = (MaxPooling2D(pool_size=2))(dense1)

dense1 = (Conv2D(32,(3,3)))(dense1)
dense1 = Dropout(0.3)(dense1)

dense1 = (Conv2D(64,(3,3)))(dense1)
dense1 = (MaxPooling2D(pool_size=2))(dense1)

dense1 = (Conv2D(64,(3,3)))(dense1)
dense1 = Dropout(0.3)(dense1)

dense1 = (Conv2D(128,(3,3),padding='same'))(dense1)
dense1 = (MaxPooling2D(pool_size=2))(dense1)

dense1 = (Conv2D(128,(3,3),padding='same'))(dense1)

fl1 = (Flatten())(dense1)
output1 = Dense(10,activation='softmax')(fl1)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
modelpath = './model/Sample/model/CIFAR10/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=80,callbacks=[chpoint],verbose=2,validation_split=0.03)

model.save('./model/Sample/model/CIFAR10/keras60_model.h5') # 가중치 까지 저장됨
model.save_weights('./model/Sample/model/CIFAR10/keras60_weight.h5') # 가중치만 저장됨

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=80)

print(f"loss : {loss}")
print(f"acc : {acc}") # acc : 0.7251999974250793