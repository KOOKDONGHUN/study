import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import EarlyStopping

''' 일평균 온도, 최저온도, 최고온도, 미세먼지농도, 강수량'''

# 1-0. 학습과 테스트 데이터 분리 및 스케일링
summer = pd.read_csv('./data/Seoul/Seoul_summer.csv',index_col=0)
summer = summer.drop('season',axis=1)

### x_pred 11일 부터 14일을 가지고 15일을 맞추기
x_pred = pd.read_csv('./data/Seoul/pred_temperature.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
x_pred = x_pred.drop('지점',axis=1)
x_pred.columns = ['시간', '평균기온', '최저기온', '최고기온']

x_pred = x_pred.interpolate()

x_pred = x_pred.drop('시간',axis=1)

x_pred = x_pred.values

# print(x_pred)

x_pred = x_pred[:-1,:]
y_past = x_pred[-1,:]

x_pred = x_pred.reshape(1,4,3)
y_past = y_past.reshape(1,3)

data = split_x(summer.values,5)
# print(data)
x_data = data[:,:-1,:-2]
y_data = data[:, -1,:-2]

# scaler = StandardScaler()

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data,y_data,shuffle=False,test_size=0.1)

x_train_scal = x_train_1.reshape(x_train_1.shape[0],x_train_1.shape[1]*x_train_1.shape[2])
x_test_scal = x_test_1.reshape(x_test_1.shape[0],x_test_1.shape[1]*x_test_1.shape[2])

scaler = MinMaxScaler()
x_train_scal = scaler.fit_transform(x_train_scal)
x_test_scal = scaler.transform(x_test_scal)

x_train_scal = x_train_scal.reshape(x_train_1.shape[0],x_train_1.shape[1],x_train_1.shape[2])
x_test_scal = x_test_scal.reshape(x_test_1.shape[0],x_test_1.shape[1],x_test_1.shape[2])

x_train = x_train_scal
x_test = x_test_scal

y_train = y_train_1
y_test = y_test_1

x_pred = x_pred.reshape(1,12)
x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(1,4,3)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(32,input_shape=(4,3),activation='relu'))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))
model.add(Dropout(0.7))

model.add(Dense(3))


# 3. 모델 컴파일 및 실행
els = EarlyStopping(monitor='loss',mode='auto',patience=7)

print(x_train,x_test)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(x_train,y_train,epochs=50,batch_size=16,validation_split=0.1,verbose=2,callbacks=[els])


# 4. 평가 예측
y_pred = model.predict(x_pred)

r2_value = r2_score(y_past,y_pred)

# print("res : ",res)

# print("r2 : ",r2_value)

# print(y_past,y_pred)

for i in range(len(y_pred)):
    print(f'실제값 : {y_test[i]} \t 예측값 : {y_pred[i]}')