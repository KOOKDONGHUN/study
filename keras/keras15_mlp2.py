#1. 데이터
import numpy as np

# 인풋3 아웃풋1의 다대1의 모델 
x = np.array([range(1,101),range(311,411),range(100)]).transpose()
y = np.array(range(711,811)).transpose()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split( 
    x,y,random_state = 66, shuffle=True,
    train_size=0.95
    )


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조의 기본

model = Sequential()
model.add(Dense(5,input_dim=3,activation='relu'))
model.add(Dense(100))
model.add(Dense(57))
model.add(Dense(26))
model.add(Dense(1))

#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=80, batch_size=2,
            validation_split=0.3,callbacks=[els])

model.summary()

#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=2)
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
