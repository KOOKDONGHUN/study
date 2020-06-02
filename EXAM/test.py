"""" 6/3 삼성전자 주가 맞춰보기 미니 프로젝트 """
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
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

# npy불러오기
samsung_data = np.load('./data/samsung.npy')
jinlo_data = np.load('./data/jinlo.npy')

size = 5

## 508 행 4열

# 1-1. 삼성 데이터 스플릿
temp_data = samsung_data
temp_data = split_x(temp_data,size)
sam_x_train = temp_data[:, 0:size-1]
sam_y_train = temp_data[:, size-1]

sam_scaler = MinMaxScaler()
sam_scaler.fit(sam_x_train)
sam_x_train = sam_scaler.transform(sam_x_train)

print(sam_x_train.shape) ## 505,4

sam_x_train = sam_x_train.reshape(sam_x_train.shape[0],sam_x_train.shape[1],1)
sam_x_train,sam_x_test,sam_y_train,sam_y_test = train_test_split(sam_x_train,sam_y_train,
                                                                 shuffle=False,
                                                                 train_size=500/505)
                                                        
# 1-2 진로 데이터 스플릿
temp_data = jinlo_data
temp_data = split_x(temp_data,size)
jin_x_train = temp_data[:, 0:size-1]

jin_scaler = MinMaxScaler()
jin_scaler.fit(jin_x_train)
jin_x_train = jin_scaler.transform(jin_x_train)

jin_x_train = jin_x_train.reshape(jin_x_train.shape[0],jin_x_train.shape[1],1)
jin_x_train,jin_x_test = train_test_split(jin_x_train,
                                          shuffle=False,
                                          train_size=500/505)

# 2. 모델 생성
model = load_model('./model/bestmodel_submit_51230.05.h5')
model.summary()

#4. 평가, 예측
loss = model.evaluate([sam_x_test,jin_x_test],
                    [sam_y_test],#,jin_y_test],
                    batch_size=7)

model.save(f'./model/model_try-{loss[0]}.h5') # 가중치 까지 저장됨 

# print(loss)
y1_predict= model.predict([sam_x_test,jin_x_test]) 

for i in range(5):
    print("pre1 : ",y1_predict[i],"  실제값 : " ,sam_y_test[i])

x = np.array([samsung_data[-4:]])
y = np.array([jinlo_data[-4:]])

x = sam_scaler.transform(x)
y = jin_scaler.transform(y)

x = x.reshape(1,4,1)
y = y.reshape(1,4,1)

pred = model.predict([x,y])
print("제출해야할 값 : ",pred)