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
# submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header=0,index_col=0)

print('x_train.shape : ',x_train.shape) # (1050000,5)
print('y_train.shape : ',y_train.shape) # (2800, 4)
print('test.shape : ',x_pred.shape) # (262500, 5)
# print('submission.shape : ',submission.shape) # (700, 4)

x_train = x_train.values
x_pred = x_pred.values
y_train = y_train.values

# y_train = y_train[:, 1:]

print('x_train.shape : ',x_train.shape) # (1050000,5)
print('y_train.shape : ',y_train.shape) # (2800, 4)
print('test.shape : ',x_pred.shape) # (262500, 5)
# print('submission.shape : ',submission.shape) # (700, 4)

x_train = x_train.reshape(2800,1875)
x_pred = x_pred.reshape(700,1875)

# x_train = x_train.reshape(2800,375,5)
# x_pred = x_pred.reshape(700,375,5)


# 2. model
model = Sequential()
model.add(Dense(20,input_dim=1875))
model.add(Dropout(0.2))
model.add(Dense(4))

# 3. compile, fit
model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

model.fit(x_train,y_train,epochs=200,batch_size=64,callbacks=[],verbose=2)

y_pred = model.predict(x_pred)

a = np.arange(2800,3500)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp2/sample_submission.csv', index = True, header=['X','Y','M','V'],index_label='id')