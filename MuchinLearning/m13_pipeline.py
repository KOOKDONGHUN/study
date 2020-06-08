import pandas as pd
import numpy as np

from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Conv2D,Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# 1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train,y_test = train_test_split(x,y,random_state=43,shuffle=True,test_size=0.2)

# 2. model
# model = SVC()

pipe = Pipeline([('scaler', MinMaxScaler()),('svm',SVC())])

pipe.fit(x_train,y_train)

print("acc : ",pipe.score(x_test,y_test))


