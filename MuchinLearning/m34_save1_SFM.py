'''
m29_eval1_SFM.py
m29_eval2_SFM.py
m29_eval3_SFM.py 에 save를 적용하시오.

save 이름에는 평가 지표를 첨가해서
가장 좋은 SFM용 save파일이 나오도록 할것
'''

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import os   #  os 를 import 하고

x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)


model = XGBRegressor(n_estimators=500,learning_rate=0.1,n_jobs=-1)
model.fit(x_train,y_train)

y_test1 = model.predict(x_test)
print(f"first XGB model pred score : {r2_score(y_test,y_test1)}")

# 핏 안하면 안돌아감
thres_holds = np.sort(model.feature_importances_)
# print("thres_holds : ",thres_holds)

# parameters = [{"n_estimators": [90, 100, 110],
#               "learning_rate": [0.1, 0.001, 0.01],
#               "max_depth": [3, 5, 7, 9],
#               "colsample_bytree": [0.6, 0.9, 1],
#               "colsample_bylevel": [0.6, 0.7, 0.9],
#               'n_jobs' : [-1]}  ]
# n_jobs = -1

filename = os.path.split(__file__)[1]

print(filename)

max = 0
for thresh in thres_holds:
    # selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median

    selec_x_train = selection.transform(x_train)

    selec_model = XGBRegressor()
    # selec_model = GridSearchCV(model,parameters,cv=3, n_jobs=n_jobs)
    selec_model.fit(selec_x_train,y_train)
    ''' 여기서 새로운 모델을 생성하는 것과 반복문에 들어오기전의 모델을 그대로 쓰는것과 차이가 있을까? '''
    # print(thresh)

    selec_x_test = selection.transform(x_test)
    y_pred = selec_model.predict(selec_x_test)

    score = r2_score(y_test,y_pred)
    # print(f'select model score : {score}')
    # print(f"model.feature_importances_ : {model.feature_importances_}")
    
    if max <= score:
        selec_model.save_model(f'./model/xgb_save/model_{filename}_save_{selec_x_train.shape[1]}_{np.round(score*100,2)}.dat')
        max = score
    # selec_model.save_model(f'./model/xgb_save/{__file__}_{np.round(thresh,2)}_{np.round(score*100,2)}.data')
    print(f"select model score : Thresh={np.round(thresh,2)} \t n={selec_x_train.shape[1]} \t r2={np.round(score*100,2)}")

# 메일 제목 : 아무개 **등



# model.fit(x_train,y_train, verbose=True, eval_metric='error',eval_set=[(x_train, y_train), (x_test, y_test)])
# model.fit(x_train,y_train, verbose=True, eval_metric=['rmse','logloss'],eval_set=[(x_train, y_train), (x_test, y_test)],
#                             early_stopping_rounds=500)

# selec_model.fit(x_train,y_train, verbose=True, eval_metric=['rmse','logloss'],eval_set=[(x_train, y_train), (x_test, y_test)],
#                             early_stopping_rounds=500)

# rmse, mae, logloss, error, auc  // error이 acc라고?