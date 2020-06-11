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


x_train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0,index_col=0)
y_train = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0,index_col=0)
x_pred = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0,index_col=0)

print('x_train.shape : ',x_train.shape) # (1050000,5)
print('y_train.shape : ',y_train.shape) # (2800, 4)
print('test.shape : ',x_pred.shape) # (262500, 5)

x_train = x_train.values
x_pred = x_pred.values
y_train = y_train.values

x_train = x_train.reshape(2800,375,5)
x_pred = x_pred.reshape(700,375,5)

x_train_sensor = x_train[:,:, 1:]
x_pred_sensor = x_pred[:,:, 1:]

print(x_train_sensor.shape) # 2800, 4, 4
print(x_pred_sensor.shape) # 700,4, 4

# print(x_train_sensor[0,:10,]) # 
# print(x_pred_sensor[0,:10,]) #

def pre_x_data_time(ls,sh1,sh2,sh3):
    new_x = list()
    tmp_list = list()
    for i in range(sh1): 
        for j in range(1,sh3):
            for k in range(sh2):
                tmp_list = ls[i,k,j]
                if tmp_list != 0:
                    tmp_list = ls[i,k,:]
                    new_x.append(tmp_list)
                    break
    return np.array(new_x)

def pre_x_data_sensor(ls,sh1,sh2,sh3):
    new_x = list()
    tmp_list = list()
    for i in range(sh1): 
        for j in range(sh3):
            for k in range(sh2):
                tmp_list = ls[i,k,j]
                if tmp_list != 0:
                    tmp_list = ls[i,k,j]
                    new_x.append(tmp_list)
                    break
    return np.array(new_x)


x_train = pre_x_data_time(x_train,x_train.shape[0],x_train.shape[1],x_train.shape[2])
x_pred = pre_x_data_time(x_pred,x_pred.shape[0],x_pred.shape[1],x_pred.shape[2])

x_train_sensor = pre_x_data_sensor(x_train_sensor,x_train_sensor.shape[0],x_train_sensor.shape[1],x_train_sensor.shape[2])
x_pred_sensor = pre_x_data_sensor(x_pred_sensor,x_pred_sensor.shape[0],x_pred_sensor.shape[1],x_pred_sensor.shape[2])

print()
print(x_train.shape)
print(x_pred.shape)
print(x_train_sensor.shape)
print(x_pred_sensor.shape)


x1_train = x_train.reshape(2800,4,5)
x1_pred = x_pred.reshape(700,4,5)

x2_train = x_train_sensor.reshape(2800,4)
x2_pred = x_pred_sensor.reshape(700,4)

print()
print(x1_train.shape)
print(x2_train.shape)
print(x1_pred.shape)
print(x2_pred.shape)



""" 


x_train_time = preprocessing_x_data(x_train_time,x_train_time.shape[0],x_train_time.shape[1],x_train_time.shape[2])
x_train_sensor = preprocessing_x_data(x_train_sensor,x_train_sensor.shape[0],x_train_sensor.shape[1],x_train_sensor.shape[2])
x_pred_time = preprocessing_x_data(x_pred_time,x_pred_time.shape[0],x_pred_time.shape[1],x_pred_time.shape[2])
x_pred_sensor = preprocessing_x_data(x_pred_sensor,x_pred_sensor.shape[0],x_pred_sensor.shape[1],x_pred_sensor.shape[2])

print(x_train_time)
print(x_train_sensor)
print(x_pred_time)
print(x_pred_sensor)

print(x_train_time.shape) # 3500,4,5으로 수정
print(x_train_sensor.shape) # 2800, 4, 4
print(x_pred_time.shape) # 875,4, 5
print(x_pred_sensor.shape) # 700,4, 4

def all_reshape(ndarray,sh1,sh2,sh3):
    ndarray = ndarray.reshape(sh1,sh2,sh3)

all_reshape(x_train_time,3500,4,5)
all_reshape(x_train_sensor,2800,4,4)
all_reshape(x_pred_time,875,4,5)
all_reshape(x_pred_sensor,700,4,4)

# 2. model
 input1 = Input()


# 3. compile, fit

model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

model.fit(new_x_train,y_train,epochs=30,batch_size=200,callbacks=[],verbose=2)

y_pred = model.predict(new_x_pred)

a = np.arange(2800,3500)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp2/sample_submission.csv', index = True, header=['X','Y','M','V'],index_label='id') """