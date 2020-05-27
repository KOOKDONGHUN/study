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
model.add(Conv2D(50,(2,2), input_shape=(28,28,1)))

model.add(Conv2D(50,(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same'))
model.add(Dropout(0.5))

# model.add(Conv2D(5,(2,2),strides=2))

model.add(Conv2D(50,(2,2),strides=2,padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(50,(2,2),padding='same',strides=2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.summary()


# 3. 컴파일(훈련준비),실행(훈련)
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=100,batch_size=30,callbacks=[els],validation_split=0.1,verbose=2)

from matplotlib import pyplot as plt

# plot 메소드의 개수에 따라 그려지는 선의 개수는 달라짐 
plt.plot(hist.history['loss']) # 하나만 넣으면 자동으로 y값으로 인식 x는 시간 순서로 알아서 잡음?
plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_loss']) # 이거 가능 validation_split 안한 모델이였던거 같은데?

plt.title('keras54 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=30)
# pred = model.predict()
# print(f"pred.shape : {pred.shape}")
# pred = np_utils.to_categorical(pred)
# print(f"pred : {pred}")
# pred = np.argmax(pred,axis=1)+1
# print(f"pred.shape : {pred.shape}")
# print(f"pred : {pred}")

print(f"loss : {loss}")
print(f"acc : {acc}")