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
import pandas as pd
import numpy as np

from MuchinLearning.m35_outliers3_KJICode import outliers

# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')




x = train.loc[:, 'rho':'990_dst']
train = outliers(x)

# test = test.loc[:, 'rho':'990_dst']

# y = train.loc[:, 'hhb':'na']

# x = x.dropna(axis=1)
# print(x.head(2))

# test = test.dropna(axis=1)
# print(test.head(2))












"""
for i in range(len(x.columns)):
    print(f'x : {x.columns[i]} \t test : {test.columns[i]}')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,random_state=5)

model = XGBRegressor()

name_ls = ['hhb','hbo2','ca','na']
tmp_dic = dict()

for i in range(4):
    model.fit(x_train,y_train.iloc[:, i])
    test_pred = model.predict(x_test)
    r2_test = r2_score(y_test.iloc[:, i], test_pred)
    mae_test = mean_absolute_error(y_test.iloc[:, i], test_pred)

    print(r2_test, "\t",mae_test)

    true_pred = model.predict(test)
    tmp_dic[name_ls[i]] = true_pred

# # submit
# df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])
# df.to_csv('./submission_test.csv',index_label='id') """