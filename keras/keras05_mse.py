#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_pred = np.array([11,12,13])
# predict

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본
model = Sequential()

model.add(Dense(5,input_dim=1))#인풋 1개 첫 아웃풋5개 activation도 default가 있음
model.add(Dense(3)) 
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse']) # 회기방식과 분류방식 2가지 ?  # mse는 실제 값과 예측값의 차이를 평균하는것 
model.fit(x,y,epochs=150, batch_size=4,callbacks=[els]) # batch_size = 32(default)

#4. 평가, 예측
loss,mse = model.evaluate(x,y,batch_size=4) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.
print("loss : ",loss)
print("mse : ",mse)

y_pred = model.predict(x_pred) #예측값
print("y_pred : ",y_pred)

"""
 문제점 ... 

 metrics 를 acc로 했을 때 1.0이 나오는 이유는 우리는 회기 모델을 테스트 하는데 우리는 분류 모델링의 지표를 사용하였다.

 훈련하는 데이터와 평가(evaluate)하는 데이터가 같기 때문에 평가에서 적절하지 못한 적합성을 판단하게 된다.

 Note
 
  회기모델은 인풋에 대응하는(비슷한?) 아웃풋수치가 나옴 y = x 일때 인풋이 1이면 아웃풋이 1
  분류모델은 인풋에 대해 정해진(딱 떨어지는) 아웃풋이 나옴 인풋에 대한 참,거짓 등과 같은 모델링
  때문에 회기 모델에서는 평가 지표가 acc가 적절하지 않다.

 """