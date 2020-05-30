from sklearn.datasets import load_breast_cancer
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
a_data = load_breast_cancer()
print(f"data : {a_data}")
print(f"data.type : {type(a_data)}")

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
print(f"x_data : {x_data}")
print(f"x_data.shape : {x_data.shape}") # x_data.shape : (569, 30)

y_data =a_data.target
print(f"y_data : {y_data}")
print(f"y_data.shape : {y_data.shape}") # y_data.shape : (569,)

feature_names = a_data.feature_names
print(f"feature_names : {feature_names}") # 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=True,
    train_size=0.8
    )

print(f"x_train.shape : {x_train.shape}") # x_train.shape : (569, 30)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(f"y_train : {y_train}")

# 2. 모델
model = Sequential()
model.add(Dense(64,input_shape=(30,)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(2))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=3,callbacks=[],verbose=2,validation_split=0.1)

# 4. 평가, 예측

plt.figure(figsize=(10,6)) # -> 도화지의 크기? 출력되는 창의 크기인가 그래프의 크기인가 

plt.subplot(2,1,1) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras82 loss plot')
plt.plot(hist.history['loss'],marker='.', c='red',label = 'loss') 
plt.plot(hist.history['val_loss'],marker='.', c='blue',label = 'val_loss')

plt. grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras82 acc plot')

plt.plot(hist.history['val_acc'])
plt.plot(hist.history['acc'])

plt. grid()

plt.ylabel('acc')
plt.xlabel('epoch')

plt.legend(['train acc','val acc'])

plt.show()

loss,acc = model.evaluate(x_test,y_test,batch_size=3)
# print("r2 : ",r2_y_predict)
print(f"loss : {loss}") 
print(f"acc : {acc}") # acc : 