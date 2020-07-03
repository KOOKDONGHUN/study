from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input, BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

# 1. data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(f"x_train.shape : {x_train.shape}")
print(f"x_test.shape : {x_test.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {y_test.shape}")

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255.0
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255.0

# 2. model
vgg16 = VGG16(weights='imagenet' , include_top=False, input_shape=(32,32,3))


model = Sequential()
model.add(vgg16)

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
# model.add(Dropout(0.1))
model.add(Activation('relu'))

model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 컴파일(훈련준비),실행(훈련)
# 원핫 인코딩을 하지 않고 sparse_categorical_crossentopy를 쓰면 됨 // 안썻던 이유는? 쓰기 길다? 말? 방구?
model.compile(optimizer=Adam(learning_rate=1e-5),loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs=20,batch_size=32,callbacks=[],verbose=2,validation_split=0.3)


# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=32)
print(f"loss : {loss}")
print(f"acc : {acc}") # acc : 


# 그림 그리기
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10,6)) # -> 도화지의 크기? 출력되는 창의 크기인가 그래프의 크기인가 

plt.subplot(2,1,1) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras112 loss plot')
plt.plot(hist.history['loss'],marker='.', c='red',label = 'loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue',label = 'val_loss')

plt. grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras112 acc plot')

plt.plot(hist.history['val_acc'])
plt.plot(hist.history['acc'])

plt. grid()

plt.ylabel('acc')
plt.xlabel('epoch')

plt.legend(['val acc','train acc'])

plt.show()
