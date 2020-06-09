from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input,Flatten, Conv1D
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

x_train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0,index_col=0)
y_train = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0,index_col=0)
x_pred = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0,index_col=0)

print('x_train.shape : ',x_train.shape) # (1050000,5)
print('y_train.shape : ',y_train.shape) # (2800, 4)
print('x_pred.shape : ',x_pred.shape) # (262500, 5)

x_train = x_train.values
x_pred = x_pred.values
y_train = y_train.values

x_train = x_train[:, 1:]
x_pred = x_pred[:, 1:]

x_train = x_train.reshape(2800,375,4)
x_pred = x_pred.reshape(700,375,4)

# new_y_train = list()

# for i in range(2800):
#     for j in range(375):
#         new_y_train.append(y_train[i, :])

# new_y_train = np.array(new_y_train)

# print(new_y_train)
# print(new_y_train.shape)


# 2. model
model = Sequential()
model.add(Conv1D(32,4,input_shape=(375,4),activation='relu'))
# model.add(LSTM(32,input_shape=(375,4),activation='relu'))
# model.add(Dense(32,input_dim=5,activation='relu'))
# model.add(Dense(32,input_dim=(375,5),activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Flatten())
model.add(Dense(4,activation='relu'))

# 3. compile, fit
model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

# model.fit(x_train,new_y_train,epochs=1,batch_size=128,callbacks=[])#,verbose=2)
model.fit(x_train,y_train,epochs=20,batch_size=128,callbacks=[])#,verbose=2)

y_pred = model.predict(x_pred)

print(y_pred.shape)

a = np.arange(2800,3500)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp2/sample_submission.csv', index = True, header=['X','Y','M','V'],index_label='id')