from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold
import matplotlib.pyplot as plt


# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

# XGBRFRegressor??????

model = XGBRegressor(n_estimators=500,learning_rate=0.1)
# model.fit(x_train,y_train, verbose=True, eval_metric='error',eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train,y_train, verbose=True, eval_metric=['rmse','logloss'],eval_set=[(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds=500)

# rmse, mae, logloss, error, auc  // error이 acc라고?

result = model.evals_result()
print(result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print(f"r2 : {round(r2*100,2)}%")

'''earlystopping  이 시작된 시점의 값 이후의 값이 아니라 잘못된 정보 전달이였다'''
eopochs = len(result['validation_0']['logloss'])
x_axis = range(0,eopochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log loss')
plt.title('XGBoost Log loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label='Train')
ax.plot(x_axis, result['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Log rmse')
plt.title('XGBoost Log rmse')
plt.show()