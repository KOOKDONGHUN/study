from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold

# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

# XGBRFRegressor??????

model = XGBRegressor(n_estimators=5,learning_rate=0.1)
# model.fit(x_train,y_train, verbose=True, eval_metric='error',eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train,y_train, verbose=True, eval_metric='rmse',eval_set=[(x_train, y_train), (x_test, y_test)])

# rmse, mae, logloss, error, auc  // error이 acc라고?

result = model.evals_result()
print(result)

score = model.score(x_test,y_test)

print(f"r2 : {score}")
