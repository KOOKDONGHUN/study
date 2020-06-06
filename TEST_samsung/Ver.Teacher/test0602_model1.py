from keras.models import Model, load_model
from keras.layers import Dense,Input,Conv2D, Dropout
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

size = 6

# 1. 데이터

# npy불러오기
samsung = np.load('./data/samsung2.npy',allow_pickle=True)
hite = np.load('./data/hite2.npy',allow_pickle=True)

# print(samsung.shape)
# print(hite.shape)

samsung = samsung.reshape(samsung.shape[0],)

samsung = split_x(samsung,size)
print(samsung.shape)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]
print(x_sam.shape)

x_hit = hite[5:510, :]

# for i in range(hite.shape[1]):
#     hite[:, i] = (split_x(hite[:, i],size))

# 2. 모델구성
input1 = Input(shape=(5,))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,))
x2 = Dense(10)(input2)
x2 = Dense(10)(x2)

merge = concatenate([x1,x2])

output = Dense(1)(merge)

model = Model(inputs=[input1,input2],outputs=output)

model.summary()

# 3. 컴파일, 훈련
model.compile(optimizer = 'adam',loss='mse')
model.fit([x_sam,hite],y_sam,epochs=1)