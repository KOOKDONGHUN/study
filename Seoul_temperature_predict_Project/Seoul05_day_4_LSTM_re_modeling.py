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
spring = pd.read_csv('./data/Seoul/Seoul_spring.csv',index_col=0)
spring = spring.drop('season',axis=1)

summer = pd.read_csv('./data/Seoul/Seoul_summer.csv',index_col=0)
summer = summer.drop('season',axis=1)

fall = pd.read_csv('./data/Seoul/Seoul_fall.csv',index_col=0)
fall = fall.drop('season',axis=1)

winter = pd.read_csv('./data/Seoul/Seoul_winter.csv',index_col=0)
winter = winter.drop('season',axis=1)

pred_temp = pd.read_csv('./data/Seoul/pred_temperature.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
pred_temp = pred_temp.drop('지점',axis=1)
pred_temp.columns = ['시간', '평균기온', '최저기온', '최고기온']

pred_rain = pd.read_csv('./data/Seoul/pred_rain.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
pred_rain = pred_rain.drop('지점',axis=1)
pred_rain.columns = ['시간', '강수량']

pred_dust = pd.read_csv('./data/Seoul/pred_dust.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)
pred_dust = pred_dust.drop('지점번호',axis=1)
pred_dust = pred_dust.drop('지점명',axis=1)
pred_dust.columns = ['시간', '미세먼지농도']

pred_temp = pred_temp.interpolate()
pred_rain = pred_rain.interpolate()
pred_dust = pred_dust.interpolate()

pred_dust = pred_dust.fillna(method='bfill')

# print(pred_temp)
# print(pred_rain)
# print(pred_dust)

x_pred = pd.merge(pred_temp,pred_dust,on='시간')
x_pred = pd.merge(x_pred,pred_rain,on='시간')
x_pred = x_pred.drop('시간',axis=1)

x_pred = x_pred.values

# print(x_pred)

x_pred = x_pred[:-1,:]
y_past = x_pred[-1,:]

x_pred = x_pred.reshape(1,4,5)
y_past = y_past.reshape(1,5)

# print(x_pred.shape)
# print(y_past.shape)

data_ls = ['spring','summer','fall','winter']

data_dic = {'spring' : spring,
            'summer' : summer,
            'fall' : fall,
            'winter' : winter}

# for i in data_ls:
#     print(data_dic[i])

split_data_dic = dict()

for i in data_ls:
    split_data_dic[i] = split_x(data_dic[i].values,5)

# for i in data_ls:
#     print(split_data_dic[i])

x_data = dict()
y_data = dict()

for i in data_ls:
    x_data[i] = split_data_dic[i][:,:-1,:]
    y_data[i] = split_data_dic[i][:, -1,:]

scaler = StandardScaler()

x_train = dict()
y_train = dict()

x_test = dict()
y_test = dict()

for i in data_ls:
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data[i],y_data[i],shuffle=False,test_size=0.1)

    x_train_scal = x_train_1.reshape(x_train_1.shape[0],x_train_1.shape[1]*x_train_1.shape[2])
    x_test_scal = x_test_1.reshape(x_test_1.shape[0],x_test_1.shape[1]*x_test_1.shape[2])

    scaler = MinMaxScaler()
    x_train_scal = scaler.fit_transform(x_train_scal)
    x_test_scal = scaler.transform(x_test_scal)

    x_train_scal = x_train_scal.reshape(x_train_1.shape[0],x_train_1.shape[1],x_train_1.shape[2])
    x_test_scal = x_test_scal.reshape(x_test_1.shape[0],x_test_1.shape[1],x_test_1.shape[2])
    print(x_train_1.shape)

    x_train[i] = x_train_scal
    x_test[i] = x_test_scal

    y_train[i] = y_train_1
    y_test[i] = y_test_1


# 2. 모델 구성
model = Sequential()
model.add(LSTM(32,input_shape=(4,5),activation='tanh'))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))
model.add(Dropout(0.7))
model.add(Dense(32))

model.add(Dense(5))


# 3. 모델 컴파일 및 실행
els = EarlyStopping(monitor='loss',mode='auto',patience=10)

model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(x_train[data_ls[1]],y_train[data_ls[1]],epochs=100,batch_size=32,validation_split=0.1,verbose=2,callbacks=[els])


# 4. 평가 예측
y_pred = model.predict(x_pred)

r2_value = r2_score(y_past,y_pred)

# print("res : ",res)

# print("r2 : ",r2_value)

# print(y_past,y_pred)

for i in range(len(y_pred)):
    print(f'실제값 : {y_test[data_ls[1]][-5+i]} \t 예측값 : {y_pred[i]}')