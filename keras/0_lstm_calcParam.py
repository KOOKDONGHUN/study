from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
# x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
x = array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
print(x.shape[0])
print(x.shape[1])
x = x.reshape(x.shape[0],1, 5)
y = array([6,7,8,9,10])
model = Sequential()
model.add(LSTM(17,activation='relu',input_shape=(1,5)))
model.add(Dense(42))
model.add(Dense(39))
model.add(Dense(41))
model.add(Dense(1))

model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(optimizer='adam',loss = 'mse')
model.fit(x,y,epochs=200,batch_size=1,callbacks=[els]) # batch_size는? -> 귀찮아서 안썻다고함 


# 4. 테스트 
x_input = array([6,7,8,9,10])
x_input = x_input.reshape(1,1,5)
print(x_input,"\n",x_input.shape) 
yhat = model.predict(x_input)
print(yhat)


'''
    현재 우리의 데이터의 shape == (4,3,1)
    1292인 이유
     4 * { ( 인풋 + 1 ) * 아웃풋 + 아웃풋**2 }

     4를 곱하는 이유는 정확하게 모르겠으나 LSTM과정(계산법?)에서의 단계가 4라고 추측한다. # 내일 수업때 추가설명해주시면 수정함
     model.add(LSTM(17{==아웃풋}),activation='relu',input_shape=(3,1{==인풋}))
'''