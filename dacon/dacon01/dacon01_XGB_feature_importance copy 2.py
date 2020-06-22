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

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

# missing value
train = train.transpose()
test = test.transpose()

train = train.interpolate()
test = test.interpolate()

train = train.fillna(0)
test = test.fillna(0)

train = train.transpose()
test = test.transpose()


# split x, y
x = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

y = train.loc[:, 'hhb':'na']

# split train, test
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,
                                                    random_state=0)

# scalling
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

test = scaler.transform(test)

# search model parameters
parameters = { 'n_estimators': [310, 350, 390], 'max_depth': [4, 5, 6],
               'learning_rate': [0.06, 0.11], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
               'colsample_bylevel': [0.6, 0.7, 0.8] }

# name_ls ( y columns == class 4 values)
name_ls = ['hhb','hbo2','ca','na']

# final predict values (submit DataFrame)
tmp_dic = dict()

# xgb model feature importance
model = XGBRegressor(n_jobs=6)

for i in range(len(y_train.iloc[0,:])):
    model.fit(x_train,y_train.iloc[:, i])

    y_test_pred = model.predict(x_test)
    r2 = r2_score(y_test.iloc[:, i],y_test_pred)
    print(f"r2 : {r2}")
    mae = mean_absolute_error(y_test.iloc[:, i],y_test_pred)
    print(f"mae : {mae}")
    
    y_pred = model.predict(test)
    tmp_dic[name_ls[i]] = y_pred

    print(f"feature importance : {model.feature_importances_}")
    plot_importance(model)

    plt.show()

# df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])

# print(df)

# df.to_csv('./submission.csv',index_label='id')
