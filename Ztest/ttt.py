import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel



# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

x = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

# view_nan(x)

# print()
x = x.interpolate()
test = test.interpolate()

# view_nan(x)

index = x.loc[pd.isna(x[x.columns[0]]), :].index
# print(x.iloc[index,:])

y = train.loc[:, 'hhb':'na']

x = x.fillna(0)

# view_nan(test)
x_pred = test.fillna(0)


x =x.values
y = y.values
x_pred = x_pred.values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66
)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#2. feature_importance
xgb = XGBRegressor()
multi_XGB = MultiOutputRegressor(xgb)
multi_XGB.fit(x_train, y_train)

print(len(multi_XGB.estimators_))   # 4


# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)

for i in range(len(multi_XGB.estimators_)):
    threshold = np.sort(multi_XGB.estimators_[i].feature_importances_)
    print(threshold)

    for thres in threshold:
        selection = SelectFromModel(multi_XGB.estimators_[i], threshold = thres, prefit = True)
        
        parameter = {
            'n_estimators': [100, 200, 400],
            'learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.1],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'colsample_bylevel':[0.6, 0.7, 0.8, 0.9],
            'max_depth': [4, 5, 6]
        }
    
        search = GridSearchCV( XGBRegressor(), parameter, cv =5, n_jobs = -1)

        select_x_train = selection.transform(x_train)

        multi_search = MultiOutputRegressor(search)
        multi_search.fit(select_x_train, y_train)
        
        select_x_test = selection.transform(x_test)

        y_pred = multi_search.predict(select_x_test)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], score*100.0))
 
        select_x_pred = selection.transform(x_pred)
        y_predict = multi_search.predict(select_x_pred)
        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./sub_XG.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')