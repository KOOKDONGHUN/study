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

x_train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0,index_col=0)
y_train = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0,index_col=0)
x_pred = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0,index_col=0)

print('x_train.shape : ',x_train.shape) # (1050000,5)
print('y_train.shape : ',y_train.shape) # (2800, 4)
print('test.shape : ',x_pred.shape) # (262500, 5)

# x_train = x_train[~(x_train == 0).any(axis=1)]

x_train = x_train.values
x_pred = x_pred.values
y_train = y_train.values

x_train = x_train.reshape(2800,375,5)
x_pred = x_pred.reshape(700,375,5)

x_train = x_train[:,357, 1:]
x_pred = x_pred[:,357, 1:]

# t1 = 0
# t2 = 375

# tmp_list = list()
# new_x_train = list()

print('x_train : ',x_train)  
print('x_train.shape : ',x_train.shape) 
print('test.shape : ',x_pred.shape) # (262500, 5)


# for i in range(2800):
#     for j in range(375):
#         for k in range(5):

# for i in range(5):
#     for j in range(375):
#         for k in range(2800):
#             tmp_list = x_train

#     tmp_list = x_train[t1:t2, :]
#     t1 += 375
#     t2 += 375
#     new_x_train.append(tmp_list)

# print(new_x_train)
# new_x_train = np.array(new_x_train)
# print(new_x_train.shape)

# print(new_x_train[0,0,:])

# x1_train = x_train[0:375,:]
# x2_train = x_train[375:375+375,:]
# x3_train = x_train[375+375:375+375+375,:]



# 2. model
model = Sequential()
model.add(Dense(64,input_dim=4,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))

model.add(Dense(4,activation='relu'))

# 3. compile, fit
model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

model.fit(x_train,y_train,epochs=30,batch_size=64,callbacks=[],verbose=2)

y_pred = model.predict(x_pred)

a = np.arange(2800,3500)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp2/sample_submission.csv', index = True, header=['X','Y','M','V'],index_label='id')