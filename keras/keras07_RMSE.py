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
model.add(Dense(2400))
model.add(Dense(2000))
model.add(Dense(1)) 

#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse','acc'])
model.fit(x_train,y_train,epochs=200, batch_size=2,callbacks=[els])

#4. 평가, 예측
loss,mse,acc = model.evaluate(x_test,y_test,batch_size=2)

print("loss : ",loss)
print("mse : ",mse)
print("acc : ",acc)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", RMSE(y_test,y_predict))


"""
 # Question

 # Note

   r2 -> 예측 모델의 주어진 데이터에 대한 적합도를 표시하는 지표 -> 결정계수  
      -> '1'에 가까울 수록 좋음 rmse나 mse는 0에 가까울 수록 좋지만 r2는 1에 가까울수록 좋다 회기모델에서 acc 대신 r2를 사용한다

                                              mse
    r2 =      1    -      ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
                            (y_test - y_predict의 평균)제곱 의 평균


 """