'''회기일때 r2와 그냥 스코어 같은지 보고
분류일때 스코어와 에큐러시가 같은지 확인
그래서? 어떨때 어떤 모델을 쓸것인가에 대해 생각해보자'''

from sklearn.svm import SVC,LinearSVC
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

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 6, shuffle=True,
    train_size=0.95
    )

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
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model4.fit(x_train,y_train)
model6.fit(x_train,y_train)

score1 = model1.score(x_train,y_train)
score2 = model2.score(x_train,y_train)
score4 = model4.score(x_train,y_train)
score6 = model6.score(x_train,y_train)

print(score1)
print(score2)
print(score4)
print(score6)

# 분류 # score 와 accuracy_score 비교
model3.fit(x_train,y_train)
model5.fit(x_train,y_train)

score3 = model3.score(x_train,y_train)
score5 = model5.score(x_train,y_train)

print(score3)
print(score5)

# # 4. evaluate, predict
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
y_pred4 = model4.predict(x_test)
y_pred5 = model5.predict(x_test)
y_pred6 = model6.predict(x_test)

print("x_test : \n",x_test,"\npred1 values : \n",y_pred1)
print("x_test : \n",x_test,"\npred2 values : \n",y_pred2)
print("x_test : \n",x_test,"\npred3 values : \n",y_pred3)
print("x_test : \n",x_test,"\npred4 values : \n",y_pred4)
print("x_test : \n",x_test,"\npred5 values : \n",y_pred5)
print("x_test : \n",x_test,"\npred6 values : \n",y_pred6)

# acc = accuracy_score(y_test,y_pred)
# print("acc : ",acc)

# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test,y_pred)
# print("r2 : ",r2_y_predict)