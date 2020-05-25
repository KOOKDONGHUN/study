''' keras45을 복사했음 '''

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
model.add(Dense(10, name='new1'))
model.add(Dense(10, name='new2'))
model.add(Dense(10, name='new3'))
model.add(Dense(10, name='new4'))
model.add(Dense(10, name='new5'))
model.add(Dense(10, name='new6'))
model.add(Dense(10, name='new7'))
model.add(Dense(10, name='new8'))
model.add(Dense(10, name='new9'))
model.add(Dense(10, name='new10'))
model.add(Dense(10, name='new11'))
model.add(Dense(10, name='new12'))
model.add(Dense(10, name='new13'))
model.add(Dense(10, name='new14'))
model.add(Dense(10, name='new15'))
model.add(Dense(10, name='new16'))
model.add(Dense(10, name='new17'))
model.add(Dense(10, name='new18'))
model.add(Dense(1,name='output')) # 
# model.add(Dense(1,input_dim=(9,),name="output")) # 모델은 실행되지만 인풋이 적용되지는 않는듯 에러 안남

model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs=40,batch_size=1,callbacks=[els],validation_split =0.1) 

print(f"hist : {hist}") # 숨겨져있다? 안보여주는 이유는? hist만 했을때는 자료형만 보여준다?
print(hist.history.keys()) # dict_keys(['loss'])
# print(f"hist : {hist.history['loss']}") # 데이터가 많다면 이렇게 출력하고 하나씩 보는것은 어려움 때문에 그래프로 확인해보자 

from matplotlib import pyplot as plt

# plot 메소드의 개수에 따라 그려지는 선의 개수는 달라짐 
plt.plot(hist.history['loss']) # 하나만 넣으면 자동으로 y값으로 인식 x는 시간 순서로 알아서 잡음?
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_acc'])
# plt.plot(hist.history['val_loss']) # 이거 가능 validation_split 안한 모델이였던거 같은데?

plt.title('keras44 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc','val loss','val acc'])
plt.show()

#train loss만 믿지 말아라 val loss 보다 안좋은?

'''
# 4. 테스트 
x_predict = np.array([7,8,9,10])
print("x_predict.shape : ", x_predict.shape)
x_predict = x_predict.reshape(x_predict.shape[0],1)
print("x_predict.shape : ", x_predict.shape)
x_predict = x_predict.reshape(x_predict.shape[1],x_predict.shape[0],1)
print(x_predict,"\n",x_predict.shape) 
y_predict = model.predict(x_predict)
print(y_predict)'''