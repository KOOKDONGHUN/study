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


# samsung_data = pd.read_csv('./data/csv/samsung.csv',
#                             index_col = None,
#                             header=0,
#                             sep=',',
#                             encoding='CP949')

# jinlo_data = pd.read_csv('./data/csv/jinlo.csv',
#                           index_col = None,
#                           header=0,
#                           sep=',',
#                           encoding='CP949')

# # print(f"samsung_data.head() : \n{samsung_data.head()}") # 일자, 시가
# # print(f"jinlo_data.head() : \n{jinlo_data.head()}") # 일자, 시가, 고가, 저가, 종가, 거래량

# '''
# samsung_data.head() :
#            일자      시가
# 0  2020-06-02  51,000
# 1  2020-06-01  50,800
# 2  2020-05-29  50,000
# 3  2020-05-28  51,100
# 4  2020-05-27  48,950

# jinlo_data.head() :
#            일자      시가      고가      저가      종가        거래량
# 0  2020-06-02  39,000     NaN     NaN     NaN        NaN
# 1  2020-06-01  36,000  38,750  36,000  38,750  1,407,345
# 2  2020-05-29  35,900  36,750  35,900  36,000    576,566
# 3  2020-05-28  36,200  36,300  35,500  35,800    548,493
# 4  2020-05-27  35,900  36,450  35,800  36,400    373,464
# '''
# # print(f"samsung_data : \n{samsung_data}")
# # print(f"jinlo_data : \n{jinlo_data}")

# # 삼성전자와 진로의 정렬을 오름차순으로 변경
# samsung_data = samsung_data.sort_values(['일자'],ascending=['True'])
# jinlo_data = jinlo_data.sort_values(['일자'],ascending=['True'])

# # print(f"samsung_data : \n{samsung_data}")
# # print(f"jinlo_data : \n{jinlo_data}")

# # nan 제거
# samsung_data = samsung_data.dropna()
# jinlo_data = jinlo_data.dropna()

# # print(f"samsung_data : \n{samsung_data}")
# # print(f"jinlo_data : \n{jinlo_data}")

# # numpy형식으로 변환
# samsung_data = samsung_data.values
# jinlo_data = jinlo_data.values

# # print(type(samsung_data))
# # print(type(jinlo_data))

# # print(f"samsung_data : \n{samsung_data}")
# # print(f"jinlo_data : \n{jinlo_data}")

# samsung_data = samsung_data[:-1, 1]
# jinlo_data = jinlo_data[:,1:]

# # print(f"samsung_data : \n{samsung_data}")
# # print(f"samsung_data.shape : \n{samsung_data.shape}")
# # print(f"jinlo_data : \n{jinlo_data}")
# # print(f"jinlo_data.shape : \n{jinlo_data.shape}")

# # 금액의 콤마를 제거
# for i in range(len(samsung_data)):
#     samsung_data[i] = samsung_data[i].replace(',','')

# for j in range(len(jinlo_data)):
#     for i in range(5):
#         jinlo_data[j, i] = jinlo_data[j, i].replace(',','')

# samsung_data = samsung_data.astype(int)
# jinlo_data = jinlo_data.astype(int)

# print("samsung_data : \n", samsung_data)
# print("jinlo_data : \n", jinlo_data)

# # # npy파일로 저장
# np.save('./data/samsung.npy',arr=samsung_data)
# np.save('./data/jinlo.npy',arr=jinlo_data)





# npy불러오기
samsung_data = np.load('./data/samsung.npy')
jinlo_data = np.load('./data/jinlo.npy')

size = 5

# 1-1. 삼성 데이터 스플릿
temp_data = samsung_data

temp_data = split_x(temp_data,size)
# print("temp_data.type : ",type(temp_data)) # 함수의 리턴값이 넘파이형식으로 리턴해줌 
# print("temp_data : ",temp_data)
# print("temp_data.shape : ", temp_data.shape)

sam_x_train = temp_data[:, 0:size-1]
# print("sam_x_train : ", sam_x_train)
# print("sam_x_train.shape : ", sam_x_train.shape)

sam_y_train = temp_data[:, size-1]
# print("sam_y_train : ", sam_y_train)
# print("sam_y_train.shape : ", sam_y_train.shape)

sam_x_train = sam_x_train.reshape(sam_x_train.shape[0],sam_x_train.shape[1],1)
# print("sam_x_train.shape : ", sam_x_train.shape)

sam_x_train,sam_x_test,sam_y_train,sam_y_test = train_test_split(sam_x_train,sam_y_train,
                                                                 shuffle=False,
                                                                 train_size=499/504)

# print("sam_x_train.shape : ", sam_x_train.shape)     
# print("sam_y_train.shape : ", sam_y_train.shape)                                                           

# 1-2 진로 데이터 스플릿
temp_data = jinlo_data[:, 0]

temp_data = split_x(temp_data,size)
# print("temp_data.type : ",type(temp_data)) # 함수의 리턴값이 넘파이형식으로 리턴해줌 
# print("temp_data : ",temp_data)
# print("temp_data.shape : ", temp_data.shape)

jin_x_train = temp_data[:, 0:size-1]
# print("jin_x_train : ", jin_x_train)
# print("jin_x_train.shape : ", jin_x_train.shape)

jin_y_train = temp_data[:, size-1]
# print("jin_y_train : ", jin_y_train)
# print("jin_y_train.shape : ", jin_y_train.shape)

jin_x_train = jin_x_train.reshape(jin_x_train.shape[0],jin_x_train.shape[1],1)
# print("jin_x_train.shape : ", jin_x_train.shape)

jin_x_train,jin_x_test,jin_y_train,jin_y_test = train_test_split(jin_x_train,jin_y_train,
                                                                 shuffle=False,
                                                                 train_size=499/504)

# print("jin_x_train.shape : ", jin_x_train.shape)     
# print("jin_y_train.shape : ", jin_y_train.shape)  
print("jin_x_test.shape : ", jin_x_test.shape)     
print("jin_y_test.shape : ", jin_y_test.shape)
print("jin_x_test : ", jin_x_test)     
print("jin_y_test : ", jin_y_test)  

# 2. 모델 생성
input1 = Input(shape=(size-1,1))
dense1 = LSTM(50)(input1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)

input2 = Input(shape=(size-1,1))
dense2 = LSTM(50)(input2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)

merge1 = concatenate([dense1,dense2])
midel1 = Dense(92,name='e1_d1')(merge1)
midel1 = Dense(84,name='e1_d2')(midel1)
midel1 = Dense(55,name='e1_d3')(midel1)

#---------------output모델 구성 ----------------#
output1 = Dense(21,name='m1_e1_d1')(midel1)
output1 = Dense(1,name='m1_e1_outpput')(output1)

# output2 = Dense(30,name='m2_e1_d1')(midel1)
# output2 = Dense(1,name='m2_e1_outpput')(output2)

# 함수형 모델의 선언
model = Model(inputs=[input1,input2],
              outputs=[output1])#,output2])


# 3. 컴파일, 실행 
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit([sam_x_train,jin_x_train],
          [sam_y_train],#jin_y_train],
          epochs=100,
          batch_size=7,
          validation_split=0.05,
          callbacks=[els],
          verbose=2)

#4. 평가, 예측
loss = model.evaluate([sam_x_test,jin_x_test],
                    [sam_y_test],#,jin_y_test],
                    batch_size=7)
# print(loss)
y1_predict= model.predict([sam_x_test,jin_x_test]) 

print("pre1 : ",y1_predict)
print(sam_y_test)
# print("pre2 : ",y1_predict[1])