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


# 사이킷런을 이용한 원-핫-인코딩
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# encoder = LabelEncoder()
# encoder.fit(y)
# temp = encoder.transform(y).reshape(-1,1)
# onehot = OneHotEncoder()
# onehot.fit(temp)
# y = onehot.transform(temp)

'''
# 판다스를 이용한 원-핫-인코딩 
y = pd.get_dummies(y)
print(f" y : {y}")
print(f" y.shape : {y.shape}")
y = y.reshape(10,5,1) # 이건 안됨 하지만 사실 할 필요없었던거 모델에서 아웃풋을 바꿔 줬으면 됐음...ㅋ
print(f" y : {y}")
print(f" y.shape : {y.shape}")'''

y = np_utils.to_categorical(y) # -> 멘 앞에 0이 왜 있는지와 제거 미션 
'''제거 방법이야 뭐 그냥 슬라이싱 때리면 될것 같은데 왜 있는지는 모르겠네'''
print(f"y.type : {type(y)}")
y = y[:, 1:]

print(f" y : {y}")
print(f" y.shape : {y.shape}")


# 2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16,input_dim=1,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17))
model.add(Dense(17,activation='sigmoid'))
model.add(Dense(5,activation='softmax')) # activation = softmax 각 출력 클래스에 대한 확률 분포 출력, 46개의 값을 모두 더하면 1이 됨? 

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc']) # loss에 바이너리??
# model.compile(optimizer=SGD(lr=0.2),loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x,y,epochs=100,batch_size=1,callbacks=[])


from matplotlib import pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])

plt.title('keras48 loss plot')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
# plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x,y,batch_size=1)
pred = model.predict([1,2,3])
print(f"pred.shape : {pred.shape}")
# pred = np_utils.to_categorical(pred)
print(f"pred : {pred}")
pred = np.argmax(pred,axis=1)+1
print(f"pred.shape : {pred.shape}")
print(f"pred : {pred}")

print(f"loss : {loss}")
print(f"acc : {acc}")