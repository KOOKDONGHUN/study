from numpy import array, sqrt, reshape
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input,RepeatVector
from keras.layers.merge import concatenate 

# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]]) # (13,3)

y = array([4,5,6,7,8,90,10,11,12,13,50,60,70]) # 

print("x1.shape",x1.shape) # x1.shape (13, 3)
# x1 = x1.reshape(x1.shape[0],x1.shape[1],1) #ValueError: Error when checking target: expected dense_4 to have 3 dimensions, but got array with shape (13, 1)
# x1 = x1.transpose() # x1.shape (13, 3) -> x1.shape (3, 13)
# print("x1.shape",x1.shape) # 

print("y.shape : ",y.shape) # (13,)

# 2. 모델구성

# model1
input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense1 = Dense(5)(dense1)
dense1 = Dense(5)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1,outputs=output1)

model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics=['mse'])
model.fit(x1,y,epochs=3,batch_size=1,callbacks=[els],verbose=2) # 

# 4. 테스트 

x1_predict = array([55,65,75])
print("x1_predict.shape : ",x1_predict.shape) # x1_predict.shape :  (3,)
print("x1_predict : ",x1_predict)
# x1_predict = x1_predict.transpose() # ValueError: Error when checking input: expected input_1 to have shape (3,) but got array with shape (1,)
x1_predict = x1_predict.reshape(1,3)
print("x1_predict.shape : ",x1_predict.shape) # x1_predict.shape :  (1,3)
print("x1_predict : ",x1_predict)

y_test = array([[85]])
print("y_test.shape : ",y_test.shape) # 

y_predict = model.predict(x1_predict)
print(y_predict)

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
