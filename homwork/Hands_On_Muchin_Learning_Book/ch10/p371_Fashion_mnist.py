import keras

fashoin_mnist = keras.datasets.fashion_mnist

(x_train_full, y_train_full), (x_test, y_test) = fashoin_mnist.load_data()

# print(x_train_full.shape)
# print(x_train_full.dtype)
# print(y_train_full.shape)

x_valid, x_train = x_train_full[:5000]/255.0  , x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000]  , y_train_full[5000:]
x_test = x_test/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', ' Dress', 'Coat', 'Sandle', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

# print(y_train)

""" model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) # 튜플이 아니고 리스트?
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax')) """

# 이렇게 하면 레이어도 랜덤서치나 그리드 서치에 추가 할 수 있을것 같다??
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# print(model.layers)
""" [<keras.layers.core.Flatten object at 0x000001EA702E8D88>, <keras.layers.core.Dense object at 0x000001EA75F924C8>,
<keras.layers.core.Dense object at 0x000001EA75F92A88>, <keras.layers.core.Dense object at 0x000001EA75F92A48>]"""
""" hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer('dense_1') is hidden1) """

""" weights, biases = hidden1.get_weights()
print(weights)
print(weights.dtype) # float32
print(type(weights)) # numpy.ndarray
print(weights.shape) # 784, 300

print(biases.dtype)
print(biases.dtype)
print(type(biases)) # 300,
print(biases.shape) """

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=30, validation_data=(x_valid,y_valid))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.DataFrame(hist.history).plot(figsize=(8,5))
plt.grid()
plt.gca().set_ylim(0,1)
plt.show()

res = model.evaluate(x_test,y_test)
print(res)

x_new = x_test[:3]
y_proba = model.predict(x_new)
print(y_proba.round(2))

y_pred = model.predict_classes(x_new)
print(y_pred)
print(np.array(class_names)[y_pred])