from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0,index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0,index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0,index_col=0)

print('train.shape : ',train.shape) # 10000,75 : x_train, x_test
print('test.shape : ',test.shape) # 10000,71 : x_predict
print('submission.shape : ',submission.shape) # 10000, 4 : y_predict

# print(train.isnull().sum()) 

# train = train.interpolate() # ? 뭔법? -> 보간법//선형보간 # 값들 전체의 선을 그리고 그에 상응하는 값을 알아서 넣어준다? ?? 하나의 선을 긋고 ? 그럼 완전 간단한 예측 모델을 만든거네
# print(train.isnull().sum()) 
# print(train.isnull().any()) 
# 회기 

# for i in train.columns:
#     # print(i)
#     print(len(train[train[i].isnull()]))

# train = train.fillna(train.mean(),axis=0)
# train = train.fillna(method='bfill')
# test = test.fillna(method='bfill')

train = train.fillna(train.mean(),axis=0)
test = train.fillna(test.mean(),axis=0)

# print()

# for i in train.columns:
#     # print(i)
#     print(len(train[train[i].isnull()]))

train = train.values
test = test.values

print(type(train))

x_data = train[:, :-4]
test = test[:, :-4]
y_data = train[:, -4:]

print(x_data.shape)
print(y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,random_state=3,shuffle=True,test_size=0.2)

# mm = MinMaxScaler()
# x_train = mm.fit_transform(x_train)
# x_test = mm.transform(x_test)
# test = mm.transform(test)

# 2. model
model = Sequential()
model.add(Dense(128,input_dim=71,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(4,activation='relu'))

# 3. compile, fit
model.compile(optimizer='adadelta',loss = 'mse', metrics = ['mae'])

model.fit(x_train,y_train,epochs=30,batch_size=32,callbacks=[],verbose=2,validation_split=0.2)

loss = model.evaluate(x_test,y_test)

print(loss)

y_pred = model.predict(test)

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')