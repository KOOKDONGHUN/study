import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

x_data = pd.read_csv('./data/csv/winequality-white.csv',
                            index_col = None,
                            header=0,
                            sep=';',
                            encoding='CP949')

y_data = pd.read_csv('./data/csv/winequality-white.csv',
                            index_col = None,
                            header=0,
                            sep=';',
                            encoding='CP949')

# print("data.head() : \n", data.head())
# print("data.keys() : \n",data.keys())

# 'fixed acidity;"volatile acidity";"citric acid";"residual sugar";
# "chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";
# "pH";"sulphates";"alcohol";"quality"'

''' 드롭해서 새로운 변수에 넣으면 타입은 None이 되고 드롭한 데이터프레임에서 데이터가 삭제 된다. '''
x_data.drop(['quality'], axis=1, inplace=True)
# print(type(x_data))
# y_data = data.drop(data[data.keys()[i] for i in range(len(x_data.keys()))])
# print("data.keys() : \n",data.keys())

y_data.drop(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol'], axis=1, inplace=True)


'''pandas DataFrame에서는 슬라이싱할때 인덱스 숫자쓸때 iloc, 컬럼명을 쓸때는 loc'''


''' 판다스 행과 열을 변환하기 '''
# print(y_data.shape)
# y_data = y_data.T
# print(y_data.shape)
''' 판다스 행과 열을 변환하기 '''


# y_data = y_data.loc[:, 'quality'].values
# print(y_data)
# print(y_data.shape)
# y_data = y_data.reshape(-1,)
# print(y_data.shape)



x_train,x_test, y_train,y_test = train_test_split(x_data,y_data,
                                                  random_state = 66, shuffle=True,
                                                  train_size=0.9)

# print("type(x_train) : ")

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. model
# 분류
model5 = RandomForestClassifier()


# 3. excute 
# 분류 # score 와 accuracy_score 비교
model5.fit(x_train,y_train.values.ravel)  # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
                                            # 해결방법은  y_train -> y_train.values().ravel 결국 넘파이로 하는 것이 가장 깔끔하게 머신러닝을 할 수 있음 
score5 = model5.score(x_train,y_train)
print("5",score5)

y_pred = model5.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)