from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))

x_data = iris.data
y_data = iris.target

print(type(x_data))
print(type(y_data))

'''numpy 파일을 저장하는 방법'''
np.save('./data/iris_x.npy',arr=x_data)
np.save('./data/iris_y.npy',arr=y_data)

x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load))
print(x_data_load.shape)

print(type(y_data_load))
print(y_data_load.shape)