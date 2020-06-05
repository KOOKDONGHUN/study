# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

import numpy as np

# 1. data
data = load_iris()

x_data = data['data']
y_data = data['target']

# print(x_data)
# print(x_data.shape)# (150, 4)
# print(type(x_data))

# y_data = np_utils.to_categorical(y_data)

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 6, shuffle=True,
    train_size=0.95
    )

# 2. model
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1) # n_neighbors=1 -> 각 객체를 1개씩 연결하겠다? 
# # 숫자가 높아질수록 그래프의 선형은 부드러워 질 것이고 낮아 질수록 각져 질 것이다. 이 정도 까지만 이해하자 
# model = KNeighborsClassifier(n_neighbors=2) # n_neighbors=2 ->각 객체를 2개씩 연결하겠다?

# 회기 # score 와 R2 비교
model = RandomForestClassifier()
# 3. excutem
model.fit(x_train,y_train)
# 4. evaluate, predict
y_pred = model.predict(x_test)
print("x_test : \n",x_test,"\npred values : \n",y_pred)

acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_pred)
print("r2 : ",r2_y_predict)


# 분류 # score 와 accuracy_score 비교
model = RandomForestRegressor()
# 3. excute
model.fit(x_train,y_train)
# 4. evaluate, predict
y_pred = model.predict(x_test)
print("x_test : \n",x_test,"\npred values : \n",y_pred)

acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_pred)
print("r2 : ",r2_y_predict)
