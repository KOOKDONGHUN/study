''' 헷갈리기 시작한다... shape '''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras39_split import split_x

# 1. 데이터
base = np.array(range(1,101))
size = 5
print("base : ",base)

data = split_x(base,size)
print("data.type : ",type(data)) # 함수의 리턴값이 넘파이형식으로 리턴해줌 
print("data : ",data)
print("data.shape : ", data.shape)

x_data = data[:, 0:size-1]
print("x_data : ", x_data)
print("x_data.shape : ", x_data.shape)

y_data = data[:, size-1]
print("y_data : ", y_data)
print("y_data.shape : ", y_data.shape)

# x_data = x_data.reshape(x_data.shape[0],x_data.shape[1],1)
# print("x_data.shape : ", x_data.shape)

from sklearn.model_selection import train_test_split

x_train,x_predict,y_train,y_predict = train_test_split( 
    x_data, y_data, shuffle=False,
    train_size=0.94
    )

x_train,x_test,y_train,y_test = train_test_split( 
    x_train,y_train, shuffle=False,
    train_size=0.8
    )

print(f"x_train : {x_train} \n x_train count : {len(x_train)}")
print(f"x_predict : {x_predict} \n x_predict count : {len(x_predict)}")
print(f"x_test : {x_test} \n x_test count : {len(x_test)}")


''' 실습1. train, test 분리할것
    실습2. 마지막 6개의 행을 predict로 만들고 싶다.
    실습3. 8:2 = train : test 비율 
    실습4. train : validation = 8:2 '''
    
    
# 2. 모델구성
model = Sequential()
model.add(Dense(20,activation='relu',input_shape = (4, )))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(optimizer='adam',loss = 'mse')
model.fit(x_train,y_train,epochs=20,batch_size=1,callbacks=[],validation_split=0.2,verbose=2)


# 4. 테스트 
loss = model.evaluate(x_test,y_test,batch_size=1)
print(f"loss : {loss}")
pred = model.predict(x_predict)
print("x_predict : ", x_predict)
print("pred : ", pred)
