from xgboost import XGBClassifier, plot_importance, XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score,GridSearchCV
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np

train = pd.read_csv('./data/dacon/comp1/train_new.csv',index_col=0,header=0)
test = pd.read_csv('./data/dacon/comp1/test_new.csv',index_col=0,header=0)

x = train.iloc[:, :-4]
y = train.loc[:, 'hhb':'na']

# print(x)
# print(y)


# 회기 모델
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

                                                    
# n_estimators = 450
# learning_rate = 0.1
# colsample_bytree = 0.85
# colsample_bylevel = 0.9

# max_depth = 6
# n_jobs = 6

parameters = [{"n_estimators": [500],
              "learning_rate": [0.09],
              "max_depth": [7],
              "colsample_bytree": [0.8],
              "colsample_bylevel": [0.8]}]

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

model = XGBRegressor(n_jobs=6)

model = GridSearchCV(model, parameters, cv = kfold)

name_ls = ['hhb','hbo2','ca','na']
tmp_dic = dict()

for i in range(len(y_train.iloc[0,:])):
    model.fit(x_train,y_train.iloc[:, i])

    y_test_pred = model.predict(x_test)
    r2 = r2_score(y_test.iloc[:, i],y_test_pred)
    print(f"r2 : {r2}")
    mae = mean_absolute_error(y_test.iloc[:, i],y_test_pred)
    print(f"mae : {mae}")
    
    y_pred = model.predict(test)
    tmp_dic[name_ls[i]] = y_pred

    # print(f"feature importance : {model.feature_importances_}")
    # plot_importance(model)

    # plt.show()

df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])

# print(df)

df.to_csv('./submission.csv',index_label='id')