import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

# 데이터 전처리
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255

# 2. 모델구성
X = np.append(x_train, x_test, axis=0)
print(X.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit_transform(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
n_components = np.argmax(cumsum >= 0.95) + 1
print(n_components)

pca = PCA(n_components=n_components)
X = pca.fit_transform(X)

x_train, x_test = train_test_split(X, train_size=60000/70000, shuffle=False)

# model
model = Sequential()
model.add(Dense(128,input_shape=(n_components,),activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# compile, fit
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train,epochs=15,batch_size=32,validation_split=0.2)

# evaluate
loss, acc = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
print(f'acc : {acc}')