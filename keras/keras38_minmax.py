""" minmaxscale   : MinMax Scaling은 최댓값 = 1, 최솟값 = 0으로 하여
                    그 사에 값들이 있도록 하는 정규화방법이다. -> 정규화

    standardscale : 평균=0과 표준편차=1이 되도록 Scaling 하는 방법이다.
                    이는 표준화라고도 부른다.

    2가지 더 있음 정리 할해야함 , lstm, GRU 파라라미터 계산법 메일 보내기 

    정규화인지 표준화인지 알아야하고 수식은 까먹지 말고 기억할것 상당히 자주쓰임 중요하다아니고 당연하다 ㅎㅎ """

from numpy import array, sqrt, reshape
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input,RepeatVector
from keras.layers.merge import concatenate 

# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
           [100,200,300]]) # (13,3)
#예측모델의 그래프가 한쪽으로 치우친 모양이 나옴 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler # PolynomialFeatures, Normalizer, Binarizer, KernelCenterer ... 개많음

# 정규화
# minmax = MinMaxScaler()
# minmax.fit(x1) # 실행 
# x1 = minmax.transform(x1) # 변환 

# 0을 기준으로 좌우로 갈림 (표준화)
std = StandardScaler()
std.fit(x1) # (13,3)
x1 = std.transform(x1)
print("x1 : ",x1)
''' y값도 표준화나 정규화를 해줘야 하는가? ->  ㄴㄴ'''


print("x1.shape",x1.shape)
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
print("x1.shape",x1.shape) # 

y = array([4,5,6,7,8,90,10,11,12,13,5000,6000,7000,400]) # 

# 이상함 reshape를 (1,3,1)을 먼저 해주면 표준화가 안되고 
# (1,3)으로 reshape를 해주고 표준화 하고 그냥 예측하면 shape에러가 나옴 -> 표준화 할때도 shape가 중요함 
x1_test = array([55,65,75])
print("x1_test.shape : ",x1_test.shape)
x1_test = x1_test.reshape(1,3)
print("x1_test.shape : ",x1_test.shape)
print("x1_test : ",x1_test)
x1_test = std.transform(x1_test)
print("x1_test : ",x1_test)
print("x1_test.shape : ",x1_test.shape)
x1_test = x1_test.reshape(1,3,1)
print("x1_test.shape : ",x1_test.shape)




y_test = array([85])
print("y_test.shape : ",y_test.shape) # 
print("y.shape : ",y.shape) # 

# 2. 모델구성
input1 = Input(shape=(3,1))
lstm1 = LSTM(12,return_sequences=True)(input1)
lstm1 = LSTM(8,return_sequences=True)(lstm1)
lstm1 = LSTM(8)(lstm1)
dense1 = Dense(9)(lstm1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
dense1 = Dense(9)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1,outputs=output1)
model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=7, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics=['mse'])
model.fit(x1,y,epochs=200,batch_size=2,callbacks=[els],verbose=2) # 

# 4. 테스트 
y_predict = model.predict(x1_test)
print("x1_test : ",x1_test)
print("y_predict : ",y_predict)