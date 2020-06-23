from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)


model = LGBMClassifier()
model.fit(x_train,y_train, verbose=False, eval_metric=['multi_logloss','multi_error'],eval_set=[(x_train, y_train), (x_test, y_test)])

parameters = [{'colsample_bytree': [1.0], 'learning_rate': [0.1,0.2,0.3], 'max_depth': [0],
        'min_child_samples': [20], 'min_child_weight': [0.001], 'min_split_gain': [0.0, 0.1],
        'n_estimators': [100,150], 'n_jobs': [-1], 'num_leaves': [31,32,33,34], 'objective': ['multiclass'],
        'random_state': [0], 'reg_alpha': [0.0], 'silent': [True],
        'subsample': [1.0], 'subsample_for_bin': [200000], 'subsample_freq': [0]}  ]


model = LGBMClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(f'score : {score}')

thresholds = np.sort(model.feature_importances_)

max = 0
filename =__file__.split("\\")[-1]

for thres in thresholds :
    selection = SelectFromModel(model, threshold=thres, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = GridSearchCV(model, parameters, n_jobs=-1, cv=3)

    selection_model.fit(select_x_train,y_train)

    y_pred = selection_model.predict(select_x_test)

    if max <= score:
        best_model = selection_model
        max = score
        best_thresh = thres
        best_shape = select_x_train.shape[1]

    score = accuracy_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, ACC: %.2f%%" %(thres, select_x_train.shape[1], score*100.0))

    print(f'selection_model.best_estimator_ : {selection_model.best_estimator_}')
    print(f'selection_model.best_params_ : {selection_model.best_params_}\n')

import joblib

joblib.dump(best_model, f'./model/LGBM_save/model_{filename}_save_{best_shape}_{np.round(max*100,2)}.dat')