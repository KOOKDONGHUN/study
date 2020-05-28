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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)

scaler2 = StandardScaler()
scaler2.fit(x2)
x2 = scaler.transform(x2)

print("x1.shape",x1.shape)
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
print("x1.shape",x1.shape) # 

print("x2.shape",x2.shape)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
print("x2.shape",x2.shape) # 

y = array([4,5,6,7,8,90,10,11,12,13,50,60,70]) # 
x1_predict = array([55,65,75])# (3, 1)
x2_predict = array([65,75,85])
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)
y_test = array([85])
print("y_test.shape : ",y_test.shape) # 
print("y.shape : ",y.shape) # 

# 2. 모델구성

# model1
input1 = Input(shape=(3,1))
lstm1 = LSTM(8)(input1)
dense1 = Dense(10)(lstm1)
dense1 = Dense(10)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(8)(dense1)

#model2
input2 = Input(shape=(3,1))
lstm2 = LSTM(10)(input2)
dense2 = Dense(11)(lstm2)
dense2 = Dense(11)(dense2)
dense2 = Dense(11)(dense2)
dense2 = Dense(11)(dense2)

# concatenate
merge1 = concatenate([dense1,dense2])
midle1 = Dense(11)(merge1)
midle1 = Dense(11)(merge1)
midle1 = Dense(11)(merge1)

# output
output1 = Dense(15)(midle1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1,input2],outputs=output1)

model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics=['mse'])
model.fit([x1,x2],y,epochs=300,batch_size=1,callbacks=[els],verbose=2) # 


# 4. 테스트 
y_predict = model.predict([x1_predict,x2_predict])
print(y_predict)


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

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2 : ",r2_y_predict)
