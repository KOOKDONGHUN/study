#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16,17,18])
x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본
model = Sequential()

model.add(Dense(5,input_dim=1,activation='relu'))#인풋 1개 첫 아웃풋5개 activation도 default가 있음
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) # 회기방식과 분류방식 2가지 ?  # mse는 실제 값과 예측값의 차이를 평균하는것 
model.fit(x_train,y_train,epochs=300, batch_size=2,
            validation_data=(x_val,y_val)) # batch_size = 32(default)

#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=3) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.

print("loss : ",loss)
print("mse : ",mse)

# y_pred = model.predict(x_pred) #예측값
# print(y_pred)
# print("\n\n")

y_predict = model.predict(x_test)
print(y_predict)


#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test,y_predict))


#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2 : ",r2_y_predict)

"""

 # Question



 # Note

    # print("1.595802357738639e-06 < 6.595802357738639e-06 = ",1.595802357738639e-06 < 6.595802357738639e-06) True
    # print("1.595802357738639e-06 > 1.595802357738639e-13 = ",1.595802357738639e-06 > 1.595802357738639e-13) True
    # print("1.595802357738639e-06 < 6.595802357738639e-13 = ",1.595802357738639e-06 < 6.595802357738639e-13) False
    # print("1.595802357738639e-13 < 6.595802357738639e-13 = ",1.595802357738639e-13 < 6.595802357738639e-13) True
    # print("1.59557738639e-13 < 6.595802357738639e-13 = ",1.59557738639e-13 < 6.595802357738639e-13)         True
    # print("0.11 > 1.7584885515771651e-06 = ", 0.11 > 1.7584885515771651e-06)                                True 

 # homework
 


 """