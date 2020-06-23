'''
m29_eval1_SFM.py
m29_eval2_SFM.py
m29_eval3_SFM.py 에 save를 적용하시오.

save 이름에는 평가 지표를 첨가해서
가장 좋은 SFM용 save파일이 나오도록 할것
'''

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

max =0 

filename =__file__.split("\\")[-1]

for thres in thresholds :
    selection = SelectFromModel(model, threshold=thres, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = GridSearchCV(model, parameters, n_jobs=-1, cv=3)

    selection_model.fit(select_x_train,y_train)

    y_pred = selection_model.predict(select_x_test)

    score = accuracy_score(y_test,y_pred)

    if max <= score:
        best_model = selection_model
        max = score
        best_thresh = thres
        best_shape = select_x_train.shape[1]

    print("Thresh=%.3f, n=%d, ACC: %.2f%%" %(thres, select_x_train.shape[1], score*100.0))

import pickle

pickle.dump(best_model,open(f'./model/xgb_save/model_{filename}_save_{best_shape}_{np.round(max*100,2)}.dat','wb'))

print("save complete!!")

# model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", 'rb'))
# print("load complete!!")