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


samsung_data = pd.read_csv('./data/csv/samsung.csv',
                            index_col = None,
                            header=0,
                            sep=',',
                            encoding='CP949')

jinlo_data = pd.read_csv('./data/csv/jinlo.csv',
                          index_col = None,
                          header=0,
                          sep=',',
                          encoding='CP949')

samsung_data = samsung_data.sort_values(['일자'],ascending=['True'])
jinlo_data = jinlo_data.sort_values(['일자'],ascending=['True'])

jinlo_data = jinlo_data.fillna(method='ffill')

samsung_data = samsung_data.values
jinlo_data = jinlo_data.values

samsung_data = samsung_data[:509, 1]
jinlo_data = jinlo_data[:509, 1:]

# print(samsung_data)
# print(jinlo_data)

samsung_data = samsung_data.astype(str)
jinlo_data = jinlo_data.astype(str)

# 금액의 콤마를 제거
for i in range(len(samsung_data)):
    samsung_data[i] = samsung_data[i].replace(',','')

for j in range(len(jinlo_data)):
    for i in range(5):
        jinlo_data[j, i] = jinlo_data[j, i].replace(',','')

# print(samsung_data)
# print(jinlo_data)

samsung_data = samsung_data.astype(int)
jinlo_data = jinlo_data.astype(int)

# print("samsung_data : \n", samsung_data)
# print("jinlo_data : \n", jinlo_data)



# # # npy파일로 저장
# np.save('./data/samsung.npy',arr=samsung_data)
# np.save('./data/jinlo.npy',arr=jinlo_data)
# # '''
# # npy불러오기
# samsung_data = np.load('./data/samsung.npy')
# jinlo_data = np.load('./data/jinlo.npy')




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

# print(sam_x_train.shape) ## 505,4

sam_x_train = sam_x_train.reshape(sam_x_train.shape[0],sam_x_train.shape[1],1)
sam_x_train,sam_x_test,sam_y_train,sam_y_test = train_test_split(sam_x_train,sam_y_train,
                                                                 shuffle=False,
                                                                 train_size=(508-size-2)/(508-size+2))



# 1-2 진로 데이터 스플릿

jin_x_train = jinlo_data

# temp_data = jinlo_data

# jin_scaler = MinMaxScaler()
# jin_scaler.fit(temp_data)
# jin_x_train = jin_scaler.transform(temp_data)

jin_x_train = split_x(jin_x_train,size)
jin_x_train = jin_x_train[:, 0:size-1]

print(jin_x_train.shape)
print(jin_x_train)

jin_x_train = jin_x_train.reshape(jin_x_train.shape[0],jin_x_train.shape[1],5)
jin_x_train,jin_x_test = train_test_split(jin_x_train,
                                          shuffle=False,
                                          train_size=(508-size-2)/(508-size+2))

print(jin_x_test.shape)
print(jin_x_train)


'''
# 2. 모델 생성
input1 = Input(shape=(size-1,1))
dense1 = LSTM(50,activation='relu')(input1)
dense1 = Dropout(0.6)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dropout(0.6)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dropout(0.6)(dense1)
dense1 = Dense(70,activation='relu')(dense1)

input2 = Input(shape=(size-1,5))
dense2 = LSTM(30,activation='relu')(input2)
dense2 = Dropout(0.6)(dense2)
dense2 = Dense(40)(dense2)
dense2 = Dropout(0.6)(dense2)
dense2 = Dense(40)(dense2)
dense2 = Dropout(0.6)(dense2)
dense2 = Dense(50,activation='relu')(dense2)

merge1 = concatenate([dense1,dense2])
output1 = Dense(50,activation='relu')(merge1)
output1 = Dropout(0.5)(output1)
output1 = Dense(40)(output1)
output1 = Dropout(0.5)(output1)
output1 = Dense(40)(output1)
output1 = Dropout(0.5)(output1)
output1 = Dense(1)(output1)

# 함수형 모델의 선언
model = Model(inputs=[input1,input2],
              outputs=[output1])#,output2])

model.summary()

# 3. 컴파일, 실행 
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit([sam_x_train,jin_x_train],
          [sam_y_train],#jin_y_train],
          epochs=100,
          batch_size=4,
          validation_split=0.05,
          callbacks=[els],
          verbose=2)

name = "0554"

#4. 평가, 예측
loss = model.evaluate([sam_x_test,jin_x_test],
                    [sam_y_test],#,jin_y_test],
                    batch_size=4)

model.save(f'./model/model_try-{name}-{loss[0]}.h5') # 가중치 까지 저장됨 

# print(loss)
y1_predict= model.predict([sam_x_test,jin_x_test]) 

for i in range(4):
    print("pre1 : ",y1_predict[i],"  실제값 : " ,sam_y_test[i])

x = np.array([samsung_data[-(size-1):]])
y = np.array([jinlo_data[-(size-1):]])

y = y.reshape(size-1,5)

x = sam_scaler.transform(x)
y = jin_scaler.transform(y)

x = x.reshape(1,size-1,1)
y = y.reshape(1,size-1,5)
 
pred = model.predict([x,y])
print("제출해야할 값 : ",pred)'''