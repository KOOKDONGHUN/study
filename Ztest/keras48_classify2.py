import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam

# 1. 데이터
x = np.array([np.arange(1,1001,1)])
y = np.array([0,1]*500)
print(x.shape)
x = x.transpose()
x = x.reshape(x.shape[0],x.shape[1],1)

y = np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split
x, _,y, _ = train_test_split(x,y,random_state=33,shuffle=True,test_size=0.0)
# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(64, input_shape = (1,1),activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dense(2,activation = 'sigmoid'))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer=Adam(learning_rate=0.02),loss = 'binary_crossentropy', metrics = ['acc'])
hist = model.fit(x,y,epochs=64,validation_split=0.4)

# 4. 평가, 예측
loss,acc = model.evaluate(x,y)

x_pred = np.array([[1002,1003,1004,1005]] )
x_pred = x_pred.transpose()
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)

pred = model.predict(x_pred)
print(f"pred : {pred}")
pred = np.argmax(pred,axis=1)
print(f"pred : {pred}")

print(f"loss : {loss}")
print(f"acc : {acc}")