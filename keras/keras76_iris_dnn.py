from sklearn.datasets import load_iris
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# 1. 데이터
a_data = load_iris()
print(f"data : {a_data}")
print(f"data.type : {type(a_data)}")

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
print(f"x_data : {x_data}")
print(f"x_data.shape : {x_data.shape}") # 150,4

y_data =a_data.target
print(f"y_data : {y_data}")
print(f"y_data.shape : {y_data.shape}") # 150,

feature_names = a_data.feature_names
print(f"feature_names : {feature_names}") # 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=False,
    train_size=0.8
    )

print(f"x_train.shape : {x_train.shape}") # x_train.shape : (120, 4)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(f"y_train : {y_train}")

# 2.  모델

model = Sequential()
model.add(Dense(100,input_shape=(4,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)

els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=12,batch_size=110,callbacks=[],verbose=2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('keras54 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
# plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=30)

print(f"loss : {loss}")
print(f"acc : {acc}")