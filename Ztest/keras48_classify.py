import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam

# 1. 데이터
x = np.arange(1,1001,1)
y = np.array([0,1]*500)

x = x.transpose()
y = np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split
x, _,y, _ = train_test_split(x,y,random_state=33,shuffle=True,test_size=0.0)
# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_dim = 1,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(16,activation='sigmoid'))
model.add(Dense(2,activation = 'tanh'))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer=Adam(learning_rate=0.02),loss = 'binary_crossentropy', metrics = ['acc'])
hist = model.fit(x,y,epochs=64,validation_split=0.4)

# 4. 평가, 예측
loss,acc = model.evaluate(x,y)
pred = model.predict([1002,1003,1004,1005])
print(f"pred : {pred}")
pred = np.argmax(pred,axis=1)
print(f"pred : {pred}")

print(f"loss : {loss}")
print(f"acc : {acc}")