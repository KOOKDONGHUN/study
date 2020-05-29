from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

els = EarlyStopping(monitor='loss', patience=6, mode='auto')

modelpath = './model/keras70/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=f'{modelpath}', monitor='val_loss', save_best_only=True,mode='auto')

tb_hist = TensorBoard(log_dir='keras70_tensorboard',histogram_freq=0,
                      write_graph=True,write_images=True)

(x_train, y_train),(x_test, y_test) = cifar100.load_data()

print(f"x_train.shape : {x_train.shape}")
print(f"x_test.shape : {x_test.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {y_test.shape}")
# plt.imshow(x_train[0])
# plt.show()

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
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=32,callbacks=[els],verbose=2,validation_split=0.01)

loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10,6)) # -> 도화지의 크기? 출력되는 창의 크기인가 그래프의 크기인가 

plt.subplot(2,1,1) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras70 loss plot')
plt.plot(hist.history['loss'],marker='.', c='red',label = 'loss') 
plt.plot(hist.history['val_loss'],marker='.', c='blue',label = 'val_loss')

plt. grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras70 acc plot')

plt.plot(hist.history['val_acc'])
plt.plot(hist.history['acc'])

plt. grid()

plt.ylabel('acc')
plt.xlabel('epoch')

plt.legend(['train acc','val acc'])

plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=32)

print(f"loss : {loss}")
print(f"acc : {acc}") # acc : 