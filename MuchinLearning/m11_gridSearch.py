import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC # 서포트 벡터 머신

# 1. data
iris = pd.read_csv('./data/csv/iris.csv',sep=',',header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=33)

parameters = [
    {"C" : [1,10,100,1000], "kernel" : ['linear','rbf'],"gamma" : [0.001, 0.0001]},
    # {"C" : [1,10,100,1000], "kernel" : ['rbf'], "gamma" : [0.001, 0.0001]}, # 경사 하강법 러닝메이트 얼마나 짤라서 내려갈지?
    {"C" : [1,10,100,1000], "kernel" : ['sigmoid'], "gamma" : [0.001, 0.0001]}
]

kfold = KFold(n_splits=5, shuffle=True)

# SVC의 어떤 파라미터? (C, kernel, gemma)의 파라미터로 크로스 발리데이션은 kfold 처럼 하겠다
model = GridSearchCV(SVC(), parameters, cv=kfold) # Cross Validation

model.fit(x_train,y_train)

print("최적의 매개변수 : ",model.best_estimator_)
y_pred = model.predict(x_test)
print(f"최종 정답률 : {accuracy_score(y_test,y_pred)}")

''' ?? kfold를 쓰고 traintestsplit을 한다는게 뭔말이야? '''