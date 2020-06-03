'''test0602_model3.py copy'''
from keras.models import Model, load_model
from keras.layers import Dense,Input,Conv2D, Dropout,LSTM
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

std1 = StandardScaler()
std1.fit(hite) # (13,3)
hite = std1.transform(hite)

pca = PCA(n_components=1)
pca.fit(hite)

hite = pca.fit_transform(hite)

print(hite)
print(hite.shape)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print('선택할 차원 수 :', d)

samsung = samsung.reshape(samsung.shape[0],)

samsung = split_x(samsung,size)
# print(samsung)
# print(samsung.shape)

x_sam = samsung[:, :size-1]
y_sam = samsung[:, size-1]
print(x_sam.shape)

std2 = StandardScaler()
std2.fit(x_sam) # (13,3)
x_sam = std2.transform(x_sam)

x_sam = x_sam.reshape(x_sam.shape[0],x_sam.shape[1],1)

# x_hit = hite[5:, :]
x_hit = split_x(hite,size)
x_hit = x_hit[:, 0:size-1]

x_hit = x_hit.reshape(x_hit.shape[0],x_hit.shape[1],1)

print(x_hit)
print(x_hit.shape)

# for i in range(hite.shape[1]):
#     hite[:, i] = (split_x(hite[:, i],size))

# 2. 모델구성
input1 = Input(shape=(5,1))
x1 = LSTM(100,activation='relu')(input1)
x1 = Dropout(0.4)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.4)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(5,1))
x2 = LSTM(100,activation='relu')(input2)
x2 = Dropout(0.7)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(100)(x2)

merge = Concatenate(axis=-1)([x1,x2])

output = Dense(100,activation='relu')(merge)
output = Dropout(0.5)(output)
output = Dense(100)(output)
output = Dropout(0.5)(output)
output = Dense(40)(output)

output = Dense(1)(output)

model = Model(inputs=[input1,input2],outputs=output)

model.summary()

# 3. 컴파일, 훈련
model.compile(optimizer = 'adam',loss='mse')
model.fit([x_sam,x_hit],y_sam,epochs=30)

# print(std1.inverse_transform(x_sam[-1,:,:].reshape(5,1)))
# print(std2.inverse_transform(x_hit[-1,:,:].reshape(5,1)))

pred = model.predict([[x_sam[-1,:,:]],[x_hit[-1,:,:]]])
print(pred)