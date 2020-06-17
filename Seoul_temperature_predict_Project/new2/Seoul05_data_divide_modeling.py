import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout

summer = pd.read_csv('./data/Seoul2/Seoul_summer.csv')
# print(summer)

summer = summer.drop('season',axis=1)
# print(summer)

size = 5

data = summer.values

data = split_x(data,size)

# print(data.shape)
# print(data)

x_data = data[:,:-1,1:]
# print(x_data)
# print(x_data.shape)

y_data = data[:,-1,1:]
# print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.01,random_state=3)

x_train_scal = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test_scal = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train_scal)
x_test_scal = scaler.transform(x_test_scal)

# x_train_scal = x_train_scal.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
# x_test_scal = x_test_scal.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])

print(x_train_scal.shape)
print(x_test_scal.shape)

# 2. model
model = Sequential()
model.add(Dense(1024,input_dim=x_train_scal.shape[1],activation='relu'))
model.add(Dense(512))
model.add(Dense(5))

# 3. compile, fit
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(x_train_scal,y_train,validation_split=0.1,batch_size=16,epochs=30)

# 4. predict

y_pred = model.predict(x_test_scal)

for i in range(len(y_pred)):
    print(f'실제값 : {y_test[i]} \t 예측값 : {np.round(y_pred[i],1)}')

r2_res = r2_score(y_test,np.round(y_pred,1))
print(f'r2_res : {r2_res}')