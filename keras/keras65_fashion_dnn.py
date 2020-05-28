from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D , Input
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data() # 흑백이미지를 불러옴 

# 다중분류 모델에서 y값에 대한 원-핫 인코딩 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 0~ 255 사이의 x 값을 0~1사이의 값으로 바꿔줌 
x_train = x_train.reshape(60000,56,14,1).astype('float32')/255
x_test = x_test.reshape(10000,56,14,1).astype('float32')/255


# 2. 모델구성
model = Sequential()
model.add(Flatten(input_shape=(56,14,1)))
model.add(Dense(64,activation='relu'))
model.add(Dense(64))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax'))

model.summary()



# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=10,batch_size=60,callbacks=[],verbose=2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])

plt.title('keras65 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=60)

print(f"loss : {loss}")
print(f"acc : {acc}") # acc : 0.8751999735832214