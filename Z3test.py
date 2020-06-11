import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D,Conv1D,MaxPool1D,Dropout
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)
from sklearn.pipeline import Pipeline, make_pipeline

# LSTM형식인데 시계열인지 명확하지 않으면 CNN으로 해야함 시계열 형식이 375단위로 잘리기 때문
# why? 잘라서 특징을 추출하기 때문

train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header=0, index_col=0)
x_prdeict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0, index_col=0)

print(train.head())

train = train.values
x_prdeict =x_prdeict.values
print(submission.shape)
print(x_prdeict.shape)
train = train.reshape(2800,375,5)
x_prdeict = x_prdeict.reshape(700,375,5)


train_time = train[0,:,:]

# print(train[0,0,4])
# print(train_time.shape)
train1 = np.zeros((1,5),dtype=np.float64)
print(train1)
train2 = list()

# for i in range(2800):
for j in range(375):
    for k in range(4):
        if(train[0,j,k+1] != 0):
            first = j
            train1 = np.append(train1,first)
            if((train[0,j,1] !=0 ) and (train[0,j,2]!=0 )and (train[0,j,3]!=0) and (train[0,j,4]!=0)):
                train2.append(j)
                
        break 
            
# train1 = train1.reshape(5,int(train1.shape[0]/15))
print(train1)
print(train2)

print(train1[5])
print(train2[0])