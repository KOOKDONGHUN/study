# 질문 DNN 과 LSTM을 앙상블 가능?
# 질문 앙상블 할때 인풋,아웃풋이 2,2인 모델과 3,3 혹은 3,2인 모델도 가능한가?

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # 
x = x.reshape(x.shape[0], x.shape[1], 1) # 

y = array([4,5,6,7]) # 

print(x.shape) # (4, 3) -> 스칼라 4
print(y.shape) # (4, )


# 2. 모델구성
model = Sequential()
# model.add(LSTM(17,activation='relu',input_shape=(3,1))) # 
model.add(LSTM(10, input_legth=3,input_dim=1)) # 
model.add(Dense(42))
model.add(Dense(39))
model.add(Dense(41))
model.add(Dense(1)) # 아웃풋 레이어 
model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(optimizer='adam',loss = 'mse')
model.fit(x,y,epochs=200,batch_size=1,callbacks=[els]) # batch_size는? -> 귀찮아서 안썻다고함 


# 4. 테스트 
x_input = array([5,6,7])
x_input = x_input.reshape(1,3,1)
print(x_input,"\n",x_input.shape) 
yhat = model.predict(x_input)
print(yhat)