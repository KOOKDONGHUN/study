import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import random

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

model1 = autoencoder(hidden_layer_size=1) # 여기다가 n_components
model2 = autoencoder(hidden_layer_size=2) # 여기다가 n_components
model4 = autoencoder(hidden_layer_size=4) # 여기다가 n_components
model8 = autoencoder(hidden_layer_size=8) # 여기다가 n_components
model16 = autoencoder(hidden_layer_size=16) # 여기다가 n_components
model32 = autoencoder(hidden_layer_size=32) # 여기다가 n_components

model1.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936
model2.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936
model4.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936
model8.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936
model16.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936
model32.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc']) # loss: 0.0936

model1.fit(x_train,x_train,epochs=10)
model2.fit(x_train,x_train,epochs=10)
model4.fit(x_train,x_train,epochs=10)
model8.fit(x_train,x_train,epochs=10)
model16.fit(x_train,x_train,epochs=10)
model32.fit(x_train,x_train,epochs=10)

output1 = model1.predict(x_test)
output2 = model2.predict(x_test)
output4 = model4.predict(x_test)
output8 = model8.predict(x_test)
output16 = model16.predict(x_test)
output32 = model32.predict(x_test)

fig, axes = plt.subplots(7, 5, figsize=(15,15))

random_imgs = random.sample(range(output1.shape[0]), 5)
outputs = [x_test, output1, output2, output4, output8, output16, output32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()