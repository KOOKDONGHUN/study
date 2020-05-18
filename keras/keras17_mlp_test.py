#1. 데이터
import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).transpose()#.reshape(100,3)
y = np.array(range(711,811)).transpose()#.reshape(100,3)

# print(x.shape) #현재는 3행100열임...
#데이터 전처리
from sklearn.model_selection import train_test_split
# train_size=0.9로 잡으면 1,3번째 변수에 0.95만큼 나머지 자동 test_size를 잡아주면 2,4번째로 할당하고 나머지 자동 
x_train,x_test,y_train,y_test = train_test_split( 
    x,y,random_state = 66, shuffle=True,
    train_size=0.95
    # train_size=0.8, test_size=0.1    0.1은 그냥 버려짐 
    # # test_size=0.05 둘중 하나만 쓰면 나머지는 알아서 자동으로 잡히는거 같은데..
    )

# print("x_train",x_train,"\ny_train",y_train)
# print("x_test",x_test,"\ny_test",y_test)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본

model = Sequential()
model.add(Dense(5,input_dim=3,activation='relu'))#인풋 1개 첫 아웃풋5개 activation도 default가 있음
                                                    #input_dim 열의 개수 
model.add(Dense(90))
model.add(Dense(19))
model.add(Dense(20))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) # 회기방식과 분류방식 2가지 ?  # mse는 실제 값과 예측값의 차이를 평균하는것 
model.fit(x_train,y_train,epochs=80, batch_size=4,
            validation_split=0.3,verbose=2) # batch_size = 32(default)

# model.summary()

#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=4) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.
# print("loss : ",loss)
# print("mse : ",mse)


y_predict = model.predict(x_test)
# print(x_test)
print(y_test)
print(y_predict)


#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test,y_predict))


#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다. 결정계수  회기모델의 보조지표 RMSE,MSE,MAE,...
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2 : ",r2_y_predict)

"""

 # Question


 # Note

    mlp 멀티레이어 퍼셉트론 

 # homework

    r2를 0.5이하, 레이어 5개이상, 노드의 개수 10개이상, epochs30개이상, batch_size 8이하 

 """
