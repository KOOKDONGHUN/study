#1. 데이터
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5,input_dim=1))
# model.add(Dense(500))
# model.add(Dense(405))
# model.add(Dense(15))
model.add(Dense(2400))
model.add(Dense(2000))
model.add(Dense(1)) 

model.summary()
#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse']) # 회기방식과 분류방식 2가지 ?  # mse는 실제 값과 예측값의 차이를 평균하는것 
# model.fit(x_train,y_train,epochs=300, batch_size=3) # batch_size = 32(default)
model.fit(x_train,y_train,epochs=200, batch_size=2,callbacks=[els]) # batch_size = 32(default)

#4. 평가, 예측
# loss,mse = model.evaluate(x_test,y_test,batch_size=3) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.
loss,mse = model.evaluate(x_test,y_test,batch_size=2) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.

print("loss : ",loss)
print("mse : ",mse)

y_pred = model.predict(x_pred) #예측값
print("y_pred : ",y_pred)

"""
 # 질문

   1 epochs 이후 테스트를 하고 2epochs를 하는가? ㄴㄴ 모든 epochs이 돌고 테스트한다. validation test는 맞음 
   epochs 미루기  -> 학습할때 마다 달라서 그런가.. -> 그냥 내가 모델을 잘 못 짰다.

 #Note

  RMSE 평균 제곱근 오차 
  RMSE는 MSE에서 나온 값에 제곱근을 하는것
  MSE가 평균 제곱오차라면 오차가 커질수록 계산은 느려진다. 

 """