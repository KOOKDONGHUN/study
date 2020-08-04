import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

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
