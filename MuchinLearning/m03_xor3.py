''' xor의 해결 '''

# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

import numpy as np

# 1. data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

x_data = np.array(x_data)
y_data = np.array(y_data)

# 2. model
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier(n_neighbors=1) # n_neighbors=1 -> 각 객체를 1개씩 연결하겠다?
# model = KNeighborsClassifier(n_neighbors=2) # n_neighbors=2 ->각 객체를 2개씩 연결하겠다?

# 3. excute
model.fit(x_data,y_data)

# 4. evaluate, predict
x_test = x_data.copy()
y_pred = model.predict(x_test)

acc = accuracy_score([0,1,1,0],y_pred)

print(x_test,"pred values : ",y_pred)
print("acc : ",acc)


'''
# XOR 의 계산 결과 데이터
xor_Data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

# 학습을 위해 데이터와 레이블 분리하기
data = []
label = []
for row in xor_Data:
    p = row[0]
    q = row[1]
    result = row[2]

    data.append([p,q])
    label.append(result)

print(data,"\n")
print(label,"\n")
'''