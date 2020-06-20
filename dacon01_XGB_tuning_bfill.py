# 과적합 방지
# 1. 훈련 데이터양을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, plot_importance, XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import RobustScaler
from hamsu import view_nan
import pandas as pd
import numpy as np

# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

x = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

print(x.shape,"000")

# view_nan(x)

# print()
# x = x.interpolate()
# test = test.interpolate()

# view_nan(x)

# index = x.loc[pd.isna(x[x.columns[0]]), :].index
# print(x.iloc[index,:])

y = train.loc[:, 'hhb':'na']

x = x.fillna(method='bfill')
x = x.fillna(method='ffill')
# x = x.fillna(0)

# view_nan(test)
test = test.fillna(method='bfill')
test = test.fillna(method='ffill')
# test = test.fillna(0)

view_nan(x)
print()
view_nan(test)

# 회기 모델
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_estimators = 450
# learning_rate = 0.1
# colsample_bytree = 0.85
# colsample_bylevel = 0.9

n_estimators = 240
learning_rate = 0.1
colsample_bytree = 0.85
colsample_bylevel = 0.6

# n_estimators = 250
# learning_rate = 0.04
# colsample_bytree = 0.75
# colsample_bylevel = 0.7

max_depth = 5
n_jobs = -1

model = XGBRegressor(  n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel,
                        max_depth=max_depth,
                        n_jobs=n_jobs   )

model = MultiOutputRegressor(model)

model.fit(x_train,y_train)

# score = model.score(x_test,y_test)
# print(f"score : {score}")

test = scaler.transform(test)

y_test_pred = model.predict(x_test)

r2 = r2_score(y_test,y_test_pred)
print(f"r2 : {r2}")

mae = mean_absolute_error(y_test,y_test_pred)
print(f"mae : {mae}")





y_pred = model.predict(test)

submissions = pd.DataFrame(y_pred,range(10000,20000),columns=['hhb','hbo2','ca','na'])

submissions.to_csv('./submission.csv',index_label='id')

# y_pred = pd.DataFrame(y_pred,columns=['s','d','f','g'])
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv')

# print(f"feature importance : {model.feature_importances_}")

# plot_importance(model)