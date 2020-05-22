""" LSTM 레이어를 5개 이상 엮어서 Dense 모델의 결과를 이겨보아라 """

from numpy import array, sqrt, reshape
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input,RepeatVector
from keras.layers.merge import concatenate 

# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]]) # (13,3)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x1)
# x1 = scaler.transform(x1)

print("x1.shape",x1.shape)
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
print("x1.shape",x1.shape) # 

y = array([4,5,6,7,8,90,10,11,12,13,50,60,70]) # 
x1_test = array([55,65,75])

x1_test = x1_test.reshape(1,3,1)

y_test = array([85])
print("y_test.shape : ",y_test.shape) # 
print("y.shape : ",y.shape) # 

# 2. 모델구성
input1 = Input(shape=(3,1))
lstm1 = LSTM(11,return_sequences=True)(input1)
lstm1 = LSTM(9,return_sequences=True)(lstm1)
lstm1 = LSTM(8,return_sequences=True)(lstm1)
lstm1 = LSTM(8,return_sequences=True)(lstm1)
lstm1 = LSTM(8,return_sequences=True)(lstm1)
# lstm1 = LSTM(8,return_sequences=True)(lstm1)
# lstm1 = LSTM(8,return_sequences=True)(lstm1)
# lstm1 = LSTM(8,return_sequences=True)(lstm1)
# lstm1 = LSTM(8,return_sequences=True)(lstm1)
# lstm1 = LSTM(8,return_sequences=True)(lstm1)
# lstm1 = LSTM(8,return_sequences=True)(lstm1)
lstm1 = LSTM(8)(lstm1)
dense1 = Dense(11)(lstm1)
dense1 = Dense(11)(dense1)
dense1 = Dense(11)(dense1)
dense1 = Dense(11)(dense1)
dense1 = Dense(11)(dense1)
dense1 = Dense(11)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1,outputs=output1)
model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=15, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics=['mse'])
model.fit(x1,y,epochs=200,batch_size=2,callbacks=[],verbose=2) # 

# 4. 테스트 
y_predict = model.predict(x1_test)
print("x1_test : ",x1_test)
print("y_predict : ",y_predict)