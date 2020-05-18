#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_pred = np.array([11,12,13]) #머신이 모델을 따라 학습한 후 학습된 머신의 예측값(y)을 구하기 위한 x
# pred는 predict의 줄임말 변수명은 아무렇게나 써도 상관없음 

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본 Deep Nural Network
model = Sequential()

model.add(Dense(5,input_dim=1))#인풋 1개 첫 아웃풋5개 activation도 default가 있음 -> ReLU (Rectified Linear Unit)
model.add(Dense(3)) 
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) # 모델에는 회기방식과 분류방식이 있다.
                                                            # mse는 실제 값과 예측값의 차이를 모두 더하고 그 값을 평균하는 계산법 
model.fit(x,y,epochs=30, batch_size=4) # batch_size = 32(default)

#4. 평가, 예측
loss,mse = model.evaluate(x,y,batch_size=4) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc 변수로 받겠다.
print("loss : ",loss)
print("mse : ",mse)

y_pred = model.predict(x_pred) # 실제 모델링한 머신의 예측값
print("y_pred : ",y_pred)
# metrics 를 acc로 했을 때 1.0이 나오는 이유는 우리가 회기 모델을 테스트 하는데 분류 모델링을 했다? 반대인가? 
# -> 회기 모델인데 분류모델의 지표(accuracy)를 사용했다. 안돼는건 아니지만  acc를 가지고 모델의 적합성을 판단하는 것은 적절하지 못하다 