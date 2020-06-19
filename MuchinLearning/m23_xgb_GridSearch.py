# 과적합 방지
# 1. 훈련 데이터양을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt

import numpy as np

# 회기 모델
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

n_jobs = -1

parameters = [{"n_estimators": [100, 200, 300],
              "learning_rate": [0.1, 0.3, 0.001, 0.01],
              "max_depth": [4, 5, 6]},
              
              {"n_estimators": [90, 100, 110],
              "learning_rate": [0.1, 0.001, 0.01],
              "max_depth": [3, 5, 7, 9],
              "colsample_bytree": [0.6, 0.9, 1]},

              {"n_estimators": [90, 100, 110],
              "learning_rate": [0.1, 0.001, 0.01],
              "max_depth": [3, 5, 7, 9],
              "colsample_bytree": [0.6, 0.9, 1],
              "colsample_bylevel": [0.6, 0.7, 0.9]}    ]


model = GridSearchCV(XGBClassifier(),parameters,cv=3,n_jobs=n_jobs)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print(f"score : {score}")

print(f"feature estimator : {model.best_estimator_}")
print(f"feature param : {model.best_params_}") # 내가 넣은것 중에서 잘나온결과

# plot_importance(model)
# plt.show()