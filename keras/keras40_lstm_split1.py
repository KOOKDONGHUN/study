''' 헷갈리기 시작한다... shape '''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras39_split import split_x

# 1. 데이터
base = np.array(range(1,11))
size = 5
print("base : ",base)

data = split_x(base,size)
print("data.type : ",type(data)) # 함수의 리턴값이 넘파이형식으로 리턴해줌 
print("data : ",data)
print("data.shape : ", data.shape)

x_train = data[:, 0:size-1]
print("x_train : ", x_train)
print("x_train.shape : ", x_train.shape)

y_train = data[:, size-1]
print("y_train : ", y_train)
print("y_train.shape : ", y_train.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
print("x_train.shape : ", x_train.shape)
'''
# 2. 모델구성
model = Sequential()
model.add(LSTM(10,activation='relu',input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(optimizer='adam',loss = 'mse')
model.fit(x_train,y_train,epochs=50,batch_size=1,callbacks=[]) 


# 4. 테스트 
x_predict = np.array([7,8,9,10])
print("x_predict.shape : ", x_predict.shape)
x_predict = x_predict.reshape(1, x_predict.shape[0])
print("x_predict.shape : ", x_predict.shape)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) 
y_predict = model.predict(x_predict)
print("x_predict : ", x_predict)
print("y_predict : ", y_predict)'''