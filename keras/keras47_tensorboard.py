''' keras46을 복사했음 '''

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

# model = load_model('./model/Save_keras44.h5')
# model.add(Dense(10, name='new1'))
# model.add(Dense(10, name='new2'))
# model.add(Dense(10, name='new3'))
# model.add(Dense(10, name='new4'))
# model.add(Dense(10, name='new5'))
# model.add(Dense(10, name='new6'))
# model.add(Dense(10, name='new7'))
# model.add(Dense(10, name='new8'))
# model.add(Dense(10, name='new9'))
# model.add(Dense(10, name='new10'))
# model.add(Dense(10, name='new11'))
# model.add(Dense(10, name='new12'))
# model.add(Dense(10, name='new13'))
# model.add(Dense(10, name='new14'))
# model.add(Dense(10, name='new15'))
# model.add(Dense(10, name='new16'))
# model.add(Dense(10, name='new17'))
# model.add(Dense(10, name='new18'))

model = Sequential()
model.add(LSTM(5,activation='relu',input_shape=(4,1)))
model.add(Dense(3))
model.add(Dense(1,name='output')) # 
# model.add(Dense(1,input_dim=(9,),name="output")) # 모델은 실행되지만 인풋이 적용되지는 않는듯 에러 안남

model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping, TensorBoard ## 추가
tb_hist = TensorBoard(log_dir='graph',histogram_freq=0,
                      write_graph=True,write_images=True)

els = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs=40,batch_size=1,callbacks=[els,tb_hist],validation_split =0.1)  ##추가

print(f"hist : {hist}") # 숨겨져있다? 안보여주는 이유는? hist만 했을때는 자료형만 보여준다?
print(hist.history.keys()) # dict_keys(['loss'])
# print(f"hist : {hist.history['loss']}") # 데이터가 많다면 이렇게 출력하고 하나씩 보는것은 어려움 때문에 그래프로 확인해보자 

'''그래프가 들어있는 경로로 이동한 후에 다음 커맨드 입력  tensorboard --logdir=. '''

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