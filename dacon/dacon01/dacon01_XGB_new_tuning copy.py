from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from hamsu import view_nan
import pandas as pd
import numpy as np


# 데이터
train = pd.read_csv('./data/dacon/comp1/new_train_nan_interpolate.csv')
test = pd.read_csv('./data/dacon/comp1/new_test_nan_interpolate.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

x = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

y = train.loc[:, 'hhb':'na']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

name_ls = ['hhb','hbo2','ca','na']
tmp_dic = dict()

# modelling
parameters = [{'max_depth': [4], # 5아직 안해봄
              'n_estimators': list(np.arange(150,450,10)),
              'learning_rate': list(np.arange(0.01,0.5,0.01)),
              'colsample_bytree':list(np.arange(0.6,0.9,0.1)),
              'colsample_bylevel': list(np.arange(0.6,0.9,0.1))}
]

n_jobs = -1

model = XGBRegressor()

model = GridSearchCV(model,parameters,cv=4, n_jobs=n_jobs)

for i in range(4):

    model.fit(x_train,y_train.iloc[:, i])

    print(f"feature estimator : {model.best_estimator_}")
    print(f"feature param : {model.best_params_}")

    y_test_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test,y_test_pred)
    r2 = r2_score(y_test,y_test_pred)

    y_pred = model.predict(test)

    # create submit DataFrame 
    tmp_dic[name_ls[i]] = y_pred

    print(f'r2 : {r2} \t mae : {mae}')

# submit
df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])
df.to_csv('./submission2.csv',index_label='id')