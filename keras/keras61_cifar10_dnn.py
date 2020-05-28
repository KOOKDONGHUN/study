from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D , Input
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(f"x_train[0] : {x_train[0]}")
print(f'y_train[0] : {y_train[0]}')

print(f"x_train.shape : {x_train.shape}")
print(f"x_test.shape : {x_test.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {y_test.shape}")

# plt.imshow(x_train[0])
# plt.show()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255



# 2. 모델구성
input1 = Input(shape=(32,32,3))

fl1 = (Flatten())(input1)
dense1 = (Dense(512))(fl1)
dense1 = Dropout(0.3)(dense1)

dense1 = (Dense(512))(dense1)
dense1 = Dropout(0.3)(dense1)

output1 = Dense(10,activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)
model.summary()



# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=5,batch_size=200,callbacks=[],verbose=2)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=200)

print(f"loss : {loss}")
print(f"acc : {acc}")