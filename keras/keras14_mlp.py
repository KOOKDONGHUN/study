#1. 데이터
import numpy as np

# 인풋3 아웃풋3의 다대다 모델
# transpose 하지 않게 되면 (3,100)의 형태 (3행,100열)
x = np.array([range(1,101),range(311,411),range(100)]).transpose()#.reshape(100,3)
y = np.array([range(101,201),range(711,811),range(100)]).transpose()#.reshape(100,3)

#    print(x.shape) #현재는 100행3열로 변환 했다. 
#    -> 행우선 열무시 이기 때문이다.
#       삼성전자 주가를 예측하기위해 100일치의 온도 데이터가 필요한게 아니라
#       온도와 날씨와 하이닉스의 주가 100일치가 필요함 말로하니까 이상한데 그림으로 하면 이해됨

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split( 
    x,y,random_state = 66, shuffle=True,
    train_size=0.95
    )


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim=3,activation='relu'))
model.add(Dense(169))
model.add(Dense(70))
model.add(Dense(71))
model.add(Dense(3))


#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=100, batch_size=3,
            validation_split=0.3)

model.summary()


#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=3)

print("loss : ",loss)
print("mse : ",mse)

y_predict = model.predict(x_test)
print(y_test)
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
 
    열 우선, 행 무시

 # homework
 


 """
