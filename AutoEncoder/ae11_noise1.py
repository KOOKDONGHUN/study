import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
    model.add(Dense(units=784,activation='sigmoid'))

    return model

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))/255
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))/255

x_train_noised = x_train + np.random.normal(0, 0.5, size=x_train.shape) # 정규분포(평균[0]에서 표준편차[0.5]만큼 값을 주겠다.)에대한 값
x_test_noised = x_test + np.random.normal(0, 0.5, size=x_test.shape)

# 정규분포이기 때문에 큰수가 더해져서 값이 음수가 생길수 있다. 때문에 다시한번 minmax를 해줘야함
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

model = autoencoder(hidden_layer_size=32) # 여기다가 n_components

# model.compile(optimizer='adam',loss='mse', metrics=['acc']) # loss: 0.0102
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936

model.fit(x_train_noised,x_train,epochs=20,validation_split=0.2)

output = model.predict(x_test_noised)

fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5,figsize=(20,7))

import random

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 노이즈 이미지를 멘 위에 그린다.
for i,ax in enumerate((ax1,ax2,ax3,ax4,ax5)):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap="gray")
    if i==0:
        ax.set_ylabel("INPUT",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아레에 그린다.
for i,ax in enumerate((ax6,ax7,ax8,ax9,ax10)):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap="gray")
    if i==0:
        ax.set_ylabel("OUTPUT",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 원본 이미지를 가장 아레에 출력
for i,ax in enumerate((ax11,ax12,ax13,ax14,ax15)):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap="gray")
    if i==0:
        ax.set_ylabel("ORIGIN",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()