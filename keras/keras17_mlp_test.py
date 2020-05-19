#1. 데이터
import numpy as np

# 다대1의 모델 r2 0.5이하 낮추기 -> 안됨 못함 어려움
x = np.array([range(1,101),range(311,411),range(100)]).transpose()
y = np.array(range(711,811)).transpose()

#데이터 전처리
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
model.add(Dense(90))
model.add(Dense(19))
model.add(Dense(20))
model.add(Dense(1))


#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=150, batch_size=4,
            validation_split=0.3,verbose=2,callbacks=[els])

# model.summary()

#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=4)

y_predict = model.predict(x_test)
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

    mlp 멀티 레이어 퍼셉트론 

 # homework

    r2를 0.5이하, 레이어 5개이상, 노드의 개수 10개이상, epochs30개이상, batch_size 8이하 
    -> 가능함? 

 """
