from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# 1. 데이터
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

# 다중분류 모델에서 y값에 대한 원-핫 인코딩 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 0~ 255 사이의 x 값을 0~1사이의 값으로 바꿔줌 
x_train = x_train.reshape(50000,32,32,3).astype('float32')/255.0
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255.0


# 2. 모델구성
input1 = Input(shape=(32,32,3))

dense1 = (Conv2D(20,(2,2),activation='elu',padding='same'))(input1)
dense1 = (Conv2D(20,(2,2),activation='elu'))(dense1)
dense1 = (MaxPooling2D(pool_size=2))(dense1)
dense1 = Dropout(0.3)(dense1)


dense1 = (Conv2D(20,(2,2),activation='elu',padding='same'))(dense1)
dense1 = (Conv2D(20,(2,2),activation='elu'))(dense1)
dense1 = (MaxPooling2D(pool_size=2))(dense1)
dense1 = Dropout(0.3)(dense1)

dense1 = (Conv2D(32,(2,2),activation='elu',padding='same'))(dense1)
dense1 = (Conv2D(32,(2,2),activation='elu'))(dense1)
dense1 = (MaxPooling2D(pool_size=2))(dense1)
dense1 = Dropout(0.4)(dense1)

fl1 = (Flatten())(dense1)


output1 = Dense(128,activation='elu')(fl1)
output1 = Dropout(0.4)(output1)

output1 = Dense(128,activation='elu')(output1)
output1 = Dropout(0.4)(output1)

output1 = Dense(100,activation='softmax')(output1)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
modelpath = './model/Sample/model/CIFAR100/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

els = EarlyStopping(monitor='loss', patience=6, mode='auto')

tb_hist = TensorBoard(log_dir='keras70_tensorboard',histogram_freq=0,
                      write_graph=True,write_images=True)

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=32,callbacks=[els,chpoint],verbose=2,validation_split=0.01)

model.save('./model/Sample/model/CIFAR100/keras70_model.h5') # 가중치 까지 저장됨
model.save_weights('./model/Sample/model/CIFAR100/keras70_weight.h5') # 가중치만 저장됨

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=32)

print(f"loss : {loss}")
print(f"acc : {acc}") # acc : 