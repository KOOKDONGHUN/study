#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10]) # shape -> (10,)
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x = np.array([1,2,3,2,3,4,3,4,5,4,5,6]) # shape -> (7,)
# print(x.shape,"\n",x)
# x = x.reshape(4,3,1)
# print(x.shape,"\n",x)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본
model = Sequential()

model.add(Dense(5,input_dim=2))#인풋 1개 첫 아웃풋5개 activation도 default가 있음
model.add(Dense(3)) #히든 레이어   #질문 -> 대괄호가 2개면 어쩌구 ...  인풋의 개수를 말하는 듯
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

# model.add(Dense(5,input_dim=1))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) #여기서 안됨을 기억하라 -> cpu와 gpu의 속도?차이를 보라는 말씀이신듯 
# model.add(Dense(1000000))
# model.add(Dense(1))

#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['acc'])
model.fit(x,y,epochs=100, batch_size=4,callbacks=[els]) #batch_size = 32(default)

#4. 평가, 예측
loss,acc = model.evaluate(x,y,batch_size=4) #evaluate -> 결과 반환(기본적으로 loss와 metrics에 있는['acc']를 반환)을 loss와 acc에 받겠다.
print("loss : ",loss)
print("acc : ",acc)