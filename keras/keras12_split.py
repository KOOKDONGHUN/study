#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

# 허접한 데이터 나누기 방법 

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]

# y_train = x[:60]
# y_val = x[60:80]
# y_test = x[80:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state = 66, shuffle=True,
    # x, y, shuffle=False,
    train_size=0.7
    )

x_val,x_test,y_val,y_test = train_test_split(
    x_test,y_test,random_state = 66, shuffle=True,
    # x_test,y_test, shuffle=False,
    train_size=0.666666666666666,
    )

print("x_train",x_train,"\ny_train",y_train)
print("x_val",x_val,"\ny_val",y_val)
print("x_test",x_test,"\ny_test",y_test)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(554))
model.add(Dense(365))
model.add(Dense(70))
model.add(Dense(1))

#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=80, batch_size=4,
            validation_data=(x_val,y_val),callbacks=[els])

#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=4)

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


 # homework
 


 """
 