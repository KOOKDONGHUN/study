# RandomizedSearch 적용

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC # 서포트 벡터 머신
# from keras.datasets import mnist
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# 1. data
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# (x_train, y_train),(x_test, y_test) = .load_data()

data = load_breast_cancer()
print(data)

x_data = data.data
y_data = data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,random_state=3,test_size=0.1)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

print(x_train,"\n", y_train,"\n",x_test,"\n", y_test,"\n")
print(type(x_train),"\n", y_train,"\n",x_test,"\n", y_test,"\n")
print(x_train.shape,"\n", y_train.shape,"\n",x_test.shape,"\n", y_test.shape,"\n")

# x_train = x_train.reshape(60000,28*28)
# x_test = x_test.reshape(10000,28*28)

# y_train = y_train.reshape(60000,)
# y_test = y_test.reshape(10000,)

parameters = {"n_estimators": list(range(10, 100, 10)),
              "max_depth": [4, 8, 12, 16],
              "max_features": [3, 5, 7, 9],
              "min_samples_split": [3, 5, 7, 9]}

kfold = KFold(n_splits=6, shuffle=True)

# SVC의 어떤 파라미터? (C, kernel, gemma)의 파라미터로 크로스 발리데이션은 kfold 처럼 하겠다
model = RandomizedSearchCV(RandomForestClassifier(),parameters,cv=kfold,n_jobs=-1) # Cross Validation

model.fit(x_train,y_train)

print("최적의 매개변수 : ",model.best_estimator_) #  best_estimator_ 과 best_parameter의 차이?
y_pred = model.predict(x_test)
print(f"최종 정답률 : {accuracy_score(y_test,y_pred)}")

''' ?? kfold를 쓰고 traintestsplit을 한다는게 뭔말이야? -> 잘못 말한듯 앞뒤가 안맞아 '''