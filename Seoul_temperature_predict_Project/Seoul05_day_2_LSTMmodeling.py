import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import EarlyStopping

''' 일평균 온도, 최저온도, 최고온도, 미세먼지농도, 강수량'''

# 1-0. 학습과 테스트 데이터 분리 및 스케일링
data = np.load('./Data/Seoul/Seoul_data.npy',allow_pickle=True)
data = data[:,1:]
print(data)
print(data.shape)

# 하루 단위로 스플릿
data = split_x(data,3)
print(data.shape)

x_data = data[:,:-1,:]
y_data = data[:,-1,:]

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,shuffle=False,test_size=0.1)

# x_train_scal = x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
# x_test_scal = x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
x_train_scal = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test_scal = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train_scal)
x_test_scal = scaler.transform(x_test_scal)

x_train_scal = x_train_scal.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
x_test_scal = x_test_scal.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])

print(x_train_scal.shape)
print(x_test_scal.shape)

print(y_train.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(32,input_shape=(2,5),activation='tanh'))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))

model.add(Dense(5))


# 3. 모델 컴파일 및 실행
els = EarlyStopping(monitor='loss',mode='auto',patience=10)

model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(x_train_scal,y_train,epochs=100,batch_size=32,validation_split=0.1,verbose=2,callbacks=[els])

# 4. 평가 예측
res = model.evaluate(x_test_scal[:-5],y_test[:-5])
x_pred = x_test_scal[-5:]
y_pred = model.predict(x_pred)

r2 = r2_score(y_test[-5:],y_pred)

print("res : ",res)

print("r2 : ",r2)

for i in range(len(y_pred)):
    print(f'실제값 : {y_test[-5+i]} \t 예측값 : {y_pred[i]}')

