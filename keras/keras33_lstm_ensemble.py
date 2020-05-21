from numpy import array, sqrt, reshape
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.layers.merge import concatenate 

# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]]) # (13,3)

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]]) # (13,3)

print("x1.shape",x1.shape)
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
print("x1.shape",x1.shape) # 

print("x2.shape",x2.shape)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
print("x2.shape",x2.shape) # 

y = array([4,5,6,7,8,90,10,11,12,13,50,60,70]) # 
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)
y_test = array([85])
print("y_test.shape : ",y_test.shape) # 
print("y.shape : ",y.shape) # 

# 2. 모델구성

# model1
input1 = Input(shape=(3,1))
lstm1 = LSTM(13)(input1)
dense1 = Dense(7)(lstm1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)

#model2
input2 = Input(shape=(3,1))
lstm2 = LSTM(9)(input2)
dense2 = Dense(7)(lstm2)
dense2 = Dense(7)(dense2)
dense2 = Dense(7)(dense2)
dense2 = Dense(6)(dense2)

# concatenate
merge1 = concatenate([dense1,dense2])
midle1 = Dense(7)(merge1)

# output
output1 = Dense(7)(midle1)
output1 = Dense(7)(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1,input2],outputs=output1)

model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=25, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics=['mse'])
model.fit([x1,x2],y,epochs=40,batch_size=1,callbacks=[],verbose=2) # 


# 4. 테스트 
y_predict = model.predict([x1_predict,x2_predict])

print("x1_predict : ",x1_predict) 
print("x2_predict : ",x2_predict)

print("y_predict : ",y_predict)

print("y_predict.shape : ",y_predict.shape)

#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return sqrt(mean_squared_error(y_test,y_predict))

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)