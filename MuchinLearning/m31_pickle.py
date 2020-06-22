from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold

# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

# model = XGBRegressor(n_estimators=5,learning_rate=0.1)
model = XGBClassifier(n_estimators=5,learning_rate=0.1)

# model.fit(x_train,y_train, verbose=True, eval_metric='error',eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train,y_train, verbose=True, eval_metric='rmse',eval_set=[(x_train, y_train), (x_test, y_test)])

# rmse, mae, logloss, error, auc  // error이 acc라고?

result = model.evals_result() # 평가? 라고 생각
# print(f'result : {result}')

y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print(f'acc1 : {acc}')

# score = model.score(x_test,y_test)
# print(f"r2 : {score}")

import pickle

pickle.dump(model,open('./model/xgb_save/cancer.pickle.dat','wb'))

print("save complete!!")

model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", 'rb'))
print("load complete!!")

y_pred = model2.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print(f'acc2 : {acc}')