'''
m29_eval1_SFM.py
m29_eval2_SFM.py
m29_eval3_SFM.py 에 save를 적용하시오.

save 이름에는 평가 지표를 첨가해서
가장 좋은 SFM용 save파일이 나오도록 할것
'''

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import os
from lightgbm import LGBMClassifier,LGBMRegressor

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

model = LGBMClassifier(n_estimators=500,learning_rate=0.1,n_jobs=-1)

model.fit(x_train,y_train)

# 핏 안하면 안돌아감
thres_holds = np.sort(model.feature_importances_)
print("thres_holds : ",thres_holds)

'''objestive = regression, binary, multiclass'''

parameters = [{'colsample_bytree': [1.0], 'learning_rate': [0.1,0.2,0.3], 'max_depth': [-1],
        'min_child_samples': [20], 'min_child_weight': [0.001], 'min_split_gain': [0.0, 0.1],
        'n_estimators': [100,150], 'n_jobs': [-1], 'num_leaves': [31,32,33,34], 'objective': ['binary'],
        'random_state': [0], 'reg_alpha': [0.0], 'silent': [True],
        'subsample': [1.0], 'subsample_for_bin': [200000], 'subsample_freq': [0]}  ]

n_jobs = -1

filename =__file__.split("\\")[-1]

print(filename)

max =0

for thresh in thres_holds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 

    selec_x_train = selection.transform(x_train)

    selec_model = GridSearchCV(model,parameters,cv=3, n_jobs=n_jobs)
    selec_model.fit(selec_x_train,y_train)

    selec_x_test = selection.transform(x_test)
    y_pred = selec_model.predict(selec_x_test)

    score = r2_score(y_test,y_pred)


    if max <= score:
        best_model = selec_model
        max = score
        best_thresh = thresh
        best_shape = selec_x_train.shape[1]

    print(f"Thresh={np.round(thresh,2)} \t n={selec_x_train.shape[1]} \t r2={np.round(score*100,2)}")

    print(f'selec_model.best_estimator_ : {selec_model.best_estimator_}')
    print(f'selec_model.best_params_ : {selec_model.best_params_}')
    

import joblib

joblib.dump(best_model, f'./model/LGBM_save/model_{filename}_save_{best_shape}_{np.round(max*100,2)}.dat')