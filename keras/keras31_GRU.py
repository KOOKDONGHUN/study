from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # 
x = x.reshape(x.shape[0], x.shape[1], 1) # 

y = array([4,5,6,7]) # 

print(x.shape) # 
print(y.shape) # 


# 2. 모델구성
model = Sequential()
model.add(GRU(17,activation='relu',input_shape=(3,1))) # 
model.add(Dense(42))
model.add(Dense(39))
model.add(Dense(41))
model.add(Dense(1)) # 아웃풋 레이어 
model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(optimizer='adam',loss = 'mse')
model.fit(x,y,epochs=200,batch_size=1,callbacks=[els]) # 


# 4. 테스트 
x_predict = array([5,6,7])
x_predict = x_predict.reshape(1,3,1)
print(x_predict,"\n",x_predict.shape) 
y_predict = model.predict(x_predict)
print(y_predict)