import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
# import pandas as pd

from keras.optimizers import SGD

# 1. 데이터
x = np.array([range(1,11)]).transpose()
y = np.array([1,2,3,4,5,1,2,3,4,5])

print(f" x : {x}")
print(f" x.shape : {x.shape}")

'''
사이킷런을 이용한 원-핫-인코딩
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
encoder.fit(y)
temp = encoder.transform(y).reshape(-1,1)
onehot = OneHotEncoder()
onehot.fit(temp)
y = onehot.transform(temp)'''

'''
y = pd.get_dummies(y)
print(f" y : {y}")
print(f" y.shape : {y.shape}")
y = y.reshape(10,5,1) # 이게 안됨 
print(f" y : {y}")
print(f" y.shape : {y.shape}")'''

y = np_utils.to_categorical(y)
print(f" y : {y}")
print(f" y.shape : {y.shape}")


# 2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30,input_dim=1,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(6,activation='sigmoid')) # activation = sigmoid? 시그모이드 함수 결과 값이 항상0,1이 나온다.
                                         # 아웃풋에 곱해주는 방법?

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
# model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc']) # loss에 바이너리??
model.compile(optimizer=SGD(lr=0.2),loss = 'binary_crossentropy', metrics = ['acc'])

hist = model.fit(x,y,epochs=100,batch_size=2,callbacks=[els])


from matplotlib import pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])

plt.title('keras48 loss plot')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x,y,batch_size=2)
pred = model.predict(x)
pred = np_utils.to_categorical(pred)
print(f"pred : {pred}")

print(f"loss : {loss}")
print(f"acc : {acc}")