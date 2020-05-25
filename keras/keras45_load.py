''' keras40을 복사했음 '''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras39_split import split_x

# 1. 데이터
base = np.array(range(1,11))
size = 4
print("base : ",base)

data = split_x(base,size)
print("data.type : ",type(data))
print("data : ",data)
print("data.shape : ", data.shape)

y_train_start_point = data.shape[1]

x_train = data[:len(data)-1]

y_train = base[y_train_start_point:]

print("x_train : ", x_train)
print("x_train.shape : ", x_train.shape)

print("y_train : ", y_train)
y_train
print("y_train.shape : ", y_train.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
print("x_train.shape : ", x_train.shape)

# LSTM 모델을 완성하시오

# 2. 모델구성
from keras.models import load_model

model = load_model('./model/Save_keras44.h5')
model.add(Dense(1,name='output')) # 
# model.add(Dense(1,input_dim=(9,),name="output")) # 모델은 실행되지만 인풋이 적용되지는 않는듯 에러 안남

model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(optimizer='adam',loss = 'mse')
model.fit(x_train,y_train,epochs=20,batch_size=1,callbacks=[]) 


# 4. 테스트 
x_predict = np.array([7,8,9,10])
print("x_predict.shape : ", x_predict.shape)
x_predict = x_predict.reshape(x_predict.shape[0],1)
print("x_predict.shape : ", x_predict.shape)
x_predict = x_predict.reshape(x_predict.shape[1],x_predict.shape[0],1)
print(x_predict,"\n",x_predict.shape) 
y_predict = model.predict(x_predict)
print(y_predict)