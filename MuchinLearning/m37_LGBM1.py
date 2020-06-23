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
import joblib

from lightgbm import LGBMClassifier,LGBMRegressor

x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)


model = LGBMRegressor(n_estimators=500,learning_rate=0.1,n_jobs=-1)
model.fit(x_train,y_train)

y_test1 = model.predict(x_test)
print(f"first XGB model pred score : {r2_score(y_test,y_test1)}")

# 핏 안하면 안돌아감
thres_holds = np.sort(model.feature_importances_)

filename = os.path.split(__file__)[1]

print(filename)

max = 0
for thresh in thres_holds:
    # selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median

    selec_x_train = selection.transform(x_train)

    selec_model = LGBMRegressor()
    # selec_model = GridSearchCV(model,parameters,cv=3, n_jobs=n_jobs)
    selec_model.fit(selec_x_train,y_train)
    ''' 여기서 새로운 모델을 생성하는 것과 반복문에 들어오기전의 모델을 그대로 쓰는것과 차이가 있을까? '''


    selec_x_test = selection.transform(x_test)
    y_pred = selec_model.predict(selec_x_test)

    score = r2_score(y_test,y_pred)
    # print(f'select model score : {score}')
    # print(f"model.feature_importances_ : {model.feature_importances_}")
    
    ### 
    if max <= score:
        best_model = selec_model
        best_thresh = thresh
        best_shape = selec_x_train.shape[1]
        # selec_model.save_model(f'./model/LGBM_save/model_{filename}_save_{selec_x_train.shape[1]}_{np.round(score*100,2)}.dat')
        max = score
    # selec_model.save_model(f'./model/xgb_save/{__file__}_{np.round(thresh,2)}_{np.round(score*100,2)}.data')
    print(f"select model score : Thresh={np.round(thresh,2)} \t n={selec_x_train.shape[1]} \t r2={np.round(score*100,2)}")

joblib.dump(best_model, f'./model/LGBM_save/model_{filename}_save_{best_shape}_{np.round(max*100,2)}.dat')