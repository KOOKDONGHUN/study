''' keras 70 copy '''

# regularizer (정규화)
# 연산값이 튀는경우 gradient vanishing
# 가중치값이 100만?

#  L1 규제 : 가중치의 절대값 합
# regukarizer.l1(l=0.01)
# L2 규제 : 가중치의 제곱 합
# regularizer.l2(l=0.01)

# 다음레이어로 전달되는 값을 줄이고 합하고

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input, BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.regularizers import l1,l2,l1_l2

els = EarlyStopping(monitor='loss', patience=6, mode='auto')

modelpath = './model/keras70/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=f'{modelpath}', monitor='val_loss', save_best_only=True,mode='auto')

tb_hist = TensorBoard(log_dir='keras70_tensorboard',histogram_freq=0,
                      write_graph=True,write_images=True)

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(f"x_train.shape : {x_train.shape}")
print(f"x_test.shape : {x_test.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {y_test.shape}")
# plt.imshow(x_train[0])
# plt.show()

# 다중분류 모델에서 y값에 대한 원-핫 인코딩 
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# 0~ 255 사이의 x 값을 0~1사이의 값으로 바꿔줌 
x_train = x_train.reshape(50000,32,32,3).astype('float32')/255.0
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255.0


# 2. 모델구성
nd = 2
ft = (2,2)
act = 'relu'
pd = 'same'

#  l2의 수를 0.001 -> 0.01 과적합이 덜해짐 0.7
#  l2의 수를 0.01 -> 0.02 비슷한듯? 0.69
#  l2의 수를 0.02 -> 0.1 과적합이 안일어난다고 봐도 무방할듯 에포가 부족해보임 // 좋아지는 속도가 천천히 좋아지는것 같다고 느껴짐 0.55
#  0.1일때 epochs 30 -> 40  :  과적합을 피했고 그래프상으로는 val_acc가 더 높게 나옴 but 최종 acc는 0.58 좋아짐  

model = Sequential()
model.add(Conv2D(nd**7,kernel_size=ft,padding=pd ,input_shape=(32,32,3),activation=act, kernel_regularizer=l2(0.1)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=ft,strides=2,padding=pd))

model.add(Conv2D(nd**8,kernel_size=ft,padding=pd,activation=act, kernel_regularizer=l2(0.1)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=ft,strides=2,padding=pd))

model.add(Conv2D(nd**9,kernel_size=ft,padding=pd,activation=act, kernel_regularizer=l2(0.1)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=ft,strides=2,padding=pd))


model.add(Flatten())

model.add(Dense(nd**8,activation=act))
model.add(Dense(nd**7,activation=act))
model.add(Dense(nd**6,activation=act))
model.add(Dense(nd**5,activation=act))
model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 컴파일(훈련준비),실행(훈련)
# 원핫 인코딩을 하지 않고 sparse_categorical_crossentopy를 쓰면 됨 // 안썻던 이유는? 쓰기 길다? 말? 방구?
model.compile(optimizer=Adam(learning_rate=1e-4),loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs=40,batch_size=32,callbacks=[els],verbose=2,validation_split=0.01)

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

