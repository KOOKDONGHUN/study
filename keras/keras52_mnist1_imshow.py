import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import mnist
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) = data

print(f"x_train[0] : {x_train[0]}")

print(f"x_train.shape : {x_train.shape}")
print(f"x_test.shape : {x_test.shape}")

print(f"y_train.shape : {y_train.shape}")
print(f"y_test.shape : {y_test.shape}")

'''plt.imshow(x_train[0],'Blues')
plt.show()'''

