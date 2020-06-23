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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

# dataset = load_breast_cancer()
# x = dataset.data
# y = dataset.target

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

# XGBRFRegressor??????

model = XGBClassifier(n_estimators=500,learning_rate=0.1,n_jobs=-1)

model.fit(x_train,y_train)

# 핏 안하면 안돌아감
thres_holds = np.sort(model.feature_importances_)
print("thres_holds : ",thres_holds)

parameters = [{"n_estimators": [90, 100, 110],
              "learning_rate": [0.1, 0.001, 0.01],
              "max_depth": [3, 5, 7, 9],
              "colsample_bytree": [0.6, 0.9, 1],
              "colsample_bylevel": [0.6, 0.7, 0.9],
              'n_jobs' : [-1]}  ]

# parameters = [{"n_estimators": [90],
#               "learning_rate": [0.1,0.2],
#               "max_depth": [3],
#               "colsample_bytree": [0.6],
#               "colsample_bylevel": [0.6],
#               'n_jobs' : [-1]}  ]

n_jobs = -1


for thresh in thres_holds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median

    selec_x_train = selection.transform(x_train)

    # print(f"selec_x_train.shape : {selec_x_train.shape}") # columns을 한개씩 줄이고 있다 

    # selec_model = XGBClassifier()
    selec_model = GridSearchCV(model,parameters,cv=3, n_jobs=n_jobs)
    selec_model.fit(selec_x_train,y_train)
    # print(thresh)

    selec_x_test = selection.transform(x_test)
    y_pred = selec_model.predict(selec_x_test)

    score = r2_score(y_test,y_pred)
    # print(score)
    # print(f"model.feature_importances_ : {model.feature_importances_}")

    print(f'selec_model.best_estimator_ : {selec_model.best_estimator_}')
    print(f'selec_model.best_params_ : {selec_model.best_params_}')
    

    print(f"Thresh={np.round(thresh,2)} \t n={selec_x_train.shape[1]} \t r2={np.round(score*100,2)}")

# 메일 제목 : 아무개 **등



# model.fit(x_train,y_train, verbose=True, eval_metric='error',eval_set=[(x_train, y_train), (x_test, y_test)])
# model.fit(x_train,y_train, verbose=True, eval_metric=['rmse','logloss'],eval_set=[(x_train, y_train), (x_test, y_test)],
#                             early_stopping_rounds=500)

selec_model.fit(x_train,y_train, verbose=True, eval_metric=['rmse','logloss'],eval_set=[(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds=500)

# rmse, mae, logloss, error, auc  // error이 acc라고?