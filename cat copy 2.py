import numpy as np
from sklearn.linear_model import Perceptron

x = np.array([range(1,11)]).transpose()
y = np.array([1,0,1,0,1,0,1,0,1,0])

print(x.shape)

model = Perceptron()
model.fit(x,y)

y_pred = model.predict(x)
print('y_pred :', y_pred)