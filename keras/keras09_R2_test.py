#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])
# predict

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본
model = Sequential()

model.add(Dense(5,input_dim=1,activation='relu'))#인풋 1개 첫 아웃풋5개 activation도 default가 있음
model.add(Dense(999))
model.add(Dense(5))
model.add(Dense(999))
model.add(Dense(5))
model.add(Dense(999))
model.add(Dense(5))
model.add(Dense(999))
model.add(Dense(5))
model.add(Dense(999))
model.add(Dense(5))
model.add(Dense(999))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse','acc']) # 회기방식과 분류방식 2가지 ?  # mse는 실제 값과 예측값의 차이를 평균하는것 
model.fit(x_train,y_train,epochs=1050, batch_size=1) # batch_size = 32(default)

#4. 평가, 예측
loss,mse,acc = model.evaluate(x_test,y_test,batch_size=1) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.
                                                      #MSE가 계산되는 시점  평가에서 나오는 mse와 모델링에서 나오는 mse가 다르다 

print("loss : ",loss)
print("mse : ",mse)
print("acc : ",acc)

y_pred = model.predict(x_pred) #예측값
print(y_pred)
print("\n\n")

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

    치우친 데이터를 범위로 나누면 나눈 범위가 같아진다?
    그래프의 면적이 같아 진다...

 # Note

    분류와 회기 수치화는 회기 비가 온다 안온다 0,1 분류방식 눈이온다면? 0,1,2 
    주가 회기만 해야하나? 분류방식으로 한다면 그룹으로 나눈다? 0원~1만(0), 1만~2만(1) ... 주가가 오르면 1 떨어지면 0

    검증데이터를 추가 하는 방법 교과서와 본사람과 교과서와 모의고사를 동시에 공부한 사람의 모의고사 점수의 차이 
    1epochs에 train하고 validation을 하고 둘다 가중치를 반영한다 fit과정 // 레이어 추가하는 거랑 비슷한 개념 

 # homework
 
    r2를 음수가 아닌 0.5이하로 줄이기.
    레이어는 인풋과 아웃풋을 포함한 5개 이상, 노드는 레이어당 각각5개이상.
    batch_size = 1  , epochs = 100 이상

    * core point 과적합을 유도하라

 """