"""" 6/3 삼성전자 주가 맞춰보기 미니 프로젝트 """
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from sklearn.decomposition import PCA
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

samsung_data = np.load('./data/samsung.npy')
jinlo_data = np.load('./data/jinlo.npy')

size = 5

# 1-1. 삼성 데이터 스플릿

sam_data = split_x(samsung_data,size)

sam_x_data = sam_data[:, 0:size-1]
# print("sam_x_data.shape : ", sam_x_data.shape) # 504,4

sam_y_data = sam_data[:, size-1]
# print("sam_y_data.shape : ", sam_y_data.shape) # 504,

scaler = MinMaxScaler()
scaler.fit(sam_x_data)

sam_x_data = scaler.transform(sam_x_data)

sam_x_data = sam_x_data.reshape(sam_x_data.shape[0],sam_x_data.shape[1],1)
# print("sam_x_data.shape : ", sam_x_data.shape)

sam_x_train,sam_x_test,sam_y_train,sam_y_test = train_test_split(sam_x_data,sam_y_data,
                                                                 shuffle=False,
                                                                 train_size=499/504)

# 1-2 진로 데이터 스플릿
temp1_data = jinlo_data[:, 0]
temp1_data = split_x(temp1_data,size)
jin_x1_data = temp1_data[:, 0:size-1]

scaler.fit(jin_x1_data)
jin_x1_data_scale = scaler.transform(jin_x1_data)

jin_x1_data_scale = jin_x1_data_scale.reshape(jin_x1_data_scale.shape[0],jin_x1_data_scale.shape[1],1)
jin_x1_train,jin_x1_test = train_test_split(jin_x1_data_scale,
                                            shuffle=False,
                                            train_size=499/504)

temp2_data = jinlo_data[:, 3]
temp2_data = split_x(temp2_data,size)
jin_x2_data = temp2_data[:, 0:size-1]

scaler.fit(jin_x2_data)
jin_x2_data_scale = scaler.transform(jin_x2_data)

jin_x2_data_scale = jin_x2_data_scale.reshape(jin_x2_data_scale.shape[0],jin_x2_data_scale.shape[1],1)
jin_x2_train,jin_x2_test = train_test_split(jin_x2_data_scale,
                                            shuffle=False,
                                            train_size=499/504)


# 2. 모델 생성
input1 = Input(shape=(size-1,1))
dense1 = LSTM(20,activation='relu')(input1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(20,activation='relu')(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(20,activation='relu')(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(20,activation='relu')(dense1)

input2 = Input(shape=(size-1,1))
dense2 = LSTM(20,activation='relu')(input2)
dense1 = Dropout(0.4)(dense1)
dense2 = Dense(20,activation='relu')(dense2)
dense1 = Dropout(0.4)(dense1)
dense2 = Dense(20,activation='relu')(dense2)
dense1 = Dropout(0.4)(dense1)
dense2 = Dense(20,activation='relu')(dense2)

input3 = Input(shape=(size-1,1))
dense3 = LSTM(20,activation='relu')(input3)
dense1 = Dropout(0.4)(dense1)
dense3 = Dense(20,activation='relu')(dense3)
dense1 = Dropout(0.4)(dense1)
dense3 = Dense(20,activation='relu')(dense3)
dense1 = Dropout(0.4)(dense1)
dense3 = Dense(20,activation='relu')(dense3)

merge1 = concatenate([dense1,dense2,dense3])
midel1 = Dense(20,activation='relu')(merge1)
dense1 = Dropout(0.4)(dense1)
midel1 = Dense(20,activation='relu')(midel1)
dense1 = Dropout(0.4)(dense1)
output1 = Dense(20,activation='relu')(midel1)
output1 = Dense(1)(output1)


# 함수형 모델의 선언
model = Model(inputs=[input1,input2,input3],
              outputs=[output1])#,output2])
model.summary()

# 3. 컴파일, 실행 
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit([sam_x_train,jin_x1_train,jin_x2_train],
          [sam_y_train],#jin_y_train],
          epochs=1,
          batch_size=4,
          validation_split=0.05,
          callbacks=[els],
          verbose=2)


# #4. 평가, 예측

# print(sam_x_test.shape)
# print(sam_x_test)
# print(jin_x1_test.shape)
# print(jin_x2_test.shape)

pred = model.predict([sam_x_test,jin_x1_test,jin_x2_test])

print("나와야함 : ",sam_y_test)
print("pre1 : ",pred)

# 36200, 35900, 36000, 39000 하이트 시가
# 35800, 36000, 38750, 38800 하이트 종가

# 51100, 50000, 50800, 51000 삼성 시가