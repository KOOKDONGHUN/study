'''회기일때 r2와 그냥 스코어 같은지 보고
분류일때 스코어와 에큐러시가 같은지 확인
그래서? 어떨때 어떤 모델을 쓸것인가에 대해 생각해보자'''

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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

# 3. excute 

# 4. evaluate, predict

def train_and_test(model):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy = round(model.score(x_test, y_test) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction

# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier())
# Random Forest
rf_pred = train_and_test(RandomForestClassifier())
# Navie Bayes
nb_pred = train_and_test(GaussianNB())