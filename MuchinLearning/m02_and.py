''' 회기 모델 '''

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

# 2. model
model = LinearSVC()

# 3. excute
model.fit(x_data,y_data)

# 4. evaluate, predict
x_test = x_data.copy()
y_pred = model.predict(x_test)

acc = accuracy_score([0,0,0,1],y_pred)

print(x_test,"pred values : ",y_pred)
print("acc : ",acc)

