''' 데이터의 y 값의 종류의 학습 데이터가 고르게 있지 않는 문제 '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 와인 데이터 읽기
wine = pd.read_csv("./data/csv/winequality-white.csv",sep=';',header=0)

data_count = wine.groupby('quality')['quality'].count() # 종류별로 모아준다고?

print(data_count)
'''
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
data_count.plot()
plt.show()

''' 분류를 축소 시킨다? '''

y_data = wine['quality']
x_data = wine.drop('quality',axis=1)

# 데이터 타입괴 shape 확인해보기
print(x_data) 
print(y_data)

# y 레이블 축소
newlist = []
for i in list(y_data):
    if i <= 4 :
        newlist += [1]
    elif i <= 7 :
        newlist += [2]
    else :
        newlist.append(3)
    # print(newlist)
# print(newlist)
y = newlist

''' 그냥 타협하는거자나 이거는 예측을 '''


x_train,x_test, y_train,y_test = train_test_split(x_data,y_data,
                                                  random_state = 66, shuffle=True,
                                                  train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. model
# 분류
model5 = RandomForestClassifier()

# 3. excute 
# 분류 # score 와 accuracy_score 비교
model5.fit(x_train,y_train)
score5 = model5.score(x_train,y_train)
print("model5 score : ",score5)

y_pred = model5.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)