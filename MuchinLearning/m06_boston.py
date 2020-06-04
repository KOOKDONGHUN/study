'''회기일때 r2와 그냥 스코어 같은지 보고
분류일때 스코어와 에큐러시가 같은지 확인
그래서? 어떨때 어떤 모델을 쓸것인가에 대해 생각해보자'''

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

import numpy as np

# 1. data
data = load_boston()

x_data = data['data']
y_data = data['target']

# print(x_data)
# print(x_data.shape)# (150, 4)
# print(type(x_data))

# print(y_data)
# print(y_data.shape)# (150, 4)
# print(type(y_data))

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=True,
    train_size=0.95
    )

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. model
# 회기
model1 = SVC()
model2 = LinearSVC()
model4 = KNeighborsRegressor()
model6 = RandomForestRegressor()

# 분류
model3 = KNeighborsClassifier()
model5 = RandomForestClassifier()

# 3. excute 
# 회기 # score 와 R2 비교
# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
model4.fit(x_train,y_train)
model6.fit(x_train,y_train)

# score1 = model1.score(x_train,y_train)
# score2 = model2.score(x_train,y_train)
score4 = model4.score(x_train,y_train)
score6 = model6.score(x_train,y_train)

# print(score1)
# print(score2)
print(score4)
print(score6)

# 분류 # score 와 accuracy_score 비교
# model3.fit(x_train,y_train)
# model5.fit(x_train,y_train)

# score3 = model3.score(x_train,y_train)
# score5 = model5.score(x_train,y_train)

# print(score3)
# print(score5)