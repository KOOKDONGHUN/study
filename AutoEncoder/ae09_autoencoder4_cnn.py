import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3), padding='same',input_shape=(28,28,1),activation='relu'))
#     model.add(MaxPooling2D(2,2))
#     model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu'))
#     model.add(MaxPooling2D((2,2))

#     model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3), padding='same',input_shape=(28,28,1),activation='relu'))
#     model.add(MaxPooling2D(2,2))
#     model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu'))
#     model.add(MaxPooling2D((2,2))

#     model.add(Conv2D(filters=1, kernel_size=(3,3), padding='same',activation='sigmoid')) ## ???

#     return model

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3), padding='same',input_shape=(28,28,1),activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=(3,3), padding='same',activation='sigmoid')) ## ???

    return model

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))/255
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))/255

model = autoencoder(hidden_layer_size=128) # 여기다가 n_components
model.summary()
""" 
model.compile(optimizer='adam',loss='mse', metrics=['acc']) # loss: 0.0102
# model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936

model.fit(x_train,x_train,epochs=2)

output = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5,figsize=(20,7))

import random

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 멘 위에 그린다.
for i,ax in enumerate((ax1,ax2,ax3,ax4,ax5)):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap="gray")
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

plt.tight_layout()
plt.show() """