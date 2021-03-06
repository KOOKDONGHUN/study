'''
m28_eval2
m28_eval3

SelectfromModel
1. 회기 m29_eval1
2. 이진분류 m29_eval2
3. 다중분류 m29_eval3

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlyStopping 적용
3. plot으로 그릴것.

4. 결과는 주석으로 소스 하단에 표시. 

5. m27 ~ 29까지 완벽히 이해할것 '''

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)


model = XGBClassifier(n_estimators=50,learning_rate=0.1,objective='multi:softmax')
model.fit(x_train,y_train, verbose=False, eval_metric=['mlogloss','merror'],eval_set=[(x_train, y_train), (x_test, y_test)])

parameters = [
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01],
    'max_depth': [4,5]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01],
    'colsample_bytree':[0.6,0.68,0.9,1],
    'max_depth': [4,5]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01],
    'colsample_bylevel': [0.6,0.68],
    'max_depth': [4,5]}
]


model = XGBClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(f'score : {score}')

thresholds = np.sort(model.feature_importances_)

for thres in thresholds :
    selection = SelectFromModel(model, threshold=thres, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = GridSearchCV(model, parameters, n_jobs=-1, cv=3)

    selection_model.fit(select_x_train,y_train)

    y_pred = selection_model.predict(select_x_test)

    score = accuracy_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, ACC: %.2f%%" %(thres, select_x_train.shape[1], score*100.0))

    print(f'selection_model.best_estimator_ : {selection_model.best_estimator_}')
    print(f'selection_model.best_params_ : {selection_model.best_params_}')