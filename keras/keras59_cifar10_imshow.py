from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(f"x_train[0] : {x_train[0]}")
print(f'y_train[0] : {y_train[0]}')

print(f"x_train.shape : {x_train.shape}")
print(f"x_test.shape : {x_test.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {y_test.shape}")

plt.imshow(x_train[0])
plt.show()