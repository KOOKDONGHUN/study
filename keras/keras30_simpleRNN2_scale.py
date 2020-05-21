from numpy import array, sqrt, reshape
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]]) # (13,3)

print("x.shape",x.shape)
x = x.reshape(x.shape[0],x.shape[1],1)
print("x.shape",x.shape) # 

y = array([4,5,6,7,8,90,10,11,12,13,50,60,70]) # 
x_predict = array([50,60,70])
x_predict = x_predict.reshape(1,3,1)
y_test = array([[80]])
print("y_test.shape : ",y_test.shape) # 
print("y.shape : ",y.shape) # 


# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(17,activation='relu',input_shape=(3,1))) # 
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(1)) # 아웃풋 레이어 
model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(optimizer='adam',loss = 'mse',metrics=['mse'])
model.fit(x,y,epochs=100,batch_size=1,callbacks=[els],verbose=2) # 


# 4. 테스트 
y_predict = model.predict(x_predict)

print("x_predict : ",x_predict) 
print("y_predict : ",y_predict)
print("y_predict.shape : ",y_predict.shape)

#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return sqrt(mean_squared_error(y_test,y_predict))

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)