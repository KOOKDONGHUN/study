import pandas as pd
import numpy as np

from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Conv2D,Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# 1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train,y_test = train_test_split(x,y,random_state=43,shuffle=True,test_size=0.2)

# gridsearchCV/randomizesearchCV에서 사용할 매개 변수
parameters = [
    {"svm__C" : [1,10,100,1000], "svm__kernel" : ['linear']},
    {"svm__C" : [1,10,100,1000], "svm__kernel" : ['rbf'], "svm__gamma" : [0.001, 0.0001]},
    {"svm__C" : [1,10,100,1000], "svm__kernel" : ['sigmoid'], "svm__gamma" : [0.001, 0.0001]}
]

# parameters = [
#     {"C" : [1,10,100,1000], "kernel" : ['linear','rbf'],"gamma" : [0.001, 0.0001]},
#     {"C" : [1,10,100,1000], "kernel" : ['rbf'], "gamma" : [0.001, 0.0001]},
#     {"C" : [1,10,100,1000], "kernel" : ['sigmoid'], "gamma" : [0.001, 0.0001]}
# ]

# 2. model
# model = SVC()

pipe = Pipeline([('scaler', MinMaxScaler()),('svm',SVC())])
# pipe = make_pipeline(MinMaxScaler(),SVC())

model = RandomizedSearchCV(pipe, parameters, cv=5)  # -> 첫번째 인자에 모델이 들어가고 파라미터, kfold

model.fit(x_train,y_train)

print("최적의 매개변수 : ",model.best_params_)
print("최적의 매개변수 : ",model.best_estimator_)
print("acc : ",model.score(x_test,y_test))

print(f"pipe.get_params():{pipe.get_params()}")

'''파이프라인의 진짜 목적? 
크로스 발리데이션할때 트랜스폼을 적절히 섞어서함 '''