''' m03_xor1.py copy '''

# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 1. data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

x_data = np.array(x_data)
y_data = np.array(y_data)

# 2. model
# model = LinearSVC()
model = SVC()

# 3. excute
model.fit(x_data,y_data)

# 4. evaluate, predict
x_test = x_data.copy()
y_pred = model.predict(x_test)

acc = accuracy_score([0,1,1,0],y_pred)

print(x_test,"pred values : ",y_pred)
print("acc : ",acc)
