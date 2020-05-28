from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D , Input
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data() # 흑백이미지를 불러옴 

'''0 : T-shirt
   1 : Trouser
   2 : Pullover
   3 : Dress
   4 : coat
   5 : sandal
   6 : Shirt
   7 : Sneaker
   8 : Bag
   9 : Ankle Boot'''

print(f"x_train[0] : {x_train[0]}")
print(f'y_train[0] : {y_train[0]}') # y_train[0] : 9

print(f"x_train.shape : {x_train.shape}") # x_train.shape : (60000, 28, 28) # 여기서 흑백인지 아닌지 안알려주면 어케알아 show에서 보면 알아?
print(f"x_test.shape : {x_test.shape}") # x_test.shape : (10000, 28, 28)
print(f"y_train.shape : {y_train.shape}") # y_train.shape : (60000,)
print(f"y_test.shape : {y_test.shape}") # y_test.shape : (10000,)

# 다중분류 모델에서 y값에 대한 원-핫 인코딩 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 0~ 255 사이의 x 값을 0~1사이의 값으로 바꿔줌 
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255


# 2. 모델구성
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32,(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64,(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(128,(2,2),padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128,(2,2),padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128,(2,2),padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(256,(2,2),padding='same'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.summary()



# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=80,callbacks=[],verbose=2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])

plt.title('keras64 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=80)

print(f"loss : {loss}")
print(f"acc : {acc}") # acc : 0.9077000021934509