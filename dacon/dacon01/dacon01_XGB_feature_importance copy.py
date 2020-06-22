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
parameters = {'colsample_bytree':list(np.arange(0.6,0.9,0.1)),
              'max_depth': [4,5,6],
              'n_estimators': list(np.arange(150,400,20)),
              'learning_rate': list(np.arange(0.01,0.5,0.05)),
              'colsample_bylevel': list(np.arange(0.6,0.9,0.1))}


# parameters = {'colsample_bytree': list(np.arange(0.6,0.9,0.1))}

# multi output def
'''
model == grid or random model
leng == y columns length
x_train == train x values
y_train == train y values
x_test == test x values
y_test == test y_values
x_pred == True predict x values

def multi_class(model, leng, x_train, y_train, x_test, y_test, x_pred):
    for i in range(leng):
        model.fit(x_train,y_train[i])
        y_test_pred = model.predict(x_test)

        r2_test_pred = r2_score(y_test, y_test_pred)
        print(f'r2_test_pred : {r2_test_pred}')

        mae_test_pred = mean_absolute_error(y_test, y_test_pred)
        print(f'mae_test_pred : {mae_test_pred}')

        y_true_pred = model.predict(x_pred)
'''



# randomsearch model

random_search_model = RandomizedSearchCV(XGBRegressor(n_jobs=6),parameters, cv=4, n_jobs=-1,n_iter=20)
for i in range(4):
    random_search_model.fit(x_train,y_train.iloc[:, i])

    random_best_estimator = random_search_model.best_estimator_
    random_best_param = random_search_model.best_params_

    print(f'{i}random_best_estimator : {random_best_estimator}')
    print(f'{i}random_best_param : {random_best_param}')

'''
0random_best_estimator : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
       colsample_bynode=1, colsample_bytree=0.8999999999999999, gamma=0,
       gpu_id=-1, importance_type='gain', interaction_constraints='',
       learning_rate=0.060000000000000005, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=nan, monotone_constraints='()',
       n_estimators=350, n_jobs=6, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)
0random_best_param : {'n_estimators': 350, 'max_depth': 6, 'learning_rate': 0.060000000000000005, 'colsample_bytree': 0.8999999999999999, 'colsample_bylevel': 0.7}     

1random_best_estimator : XGBRegressor(base_score=0.5, booster='gbtree',
       colsample_bylevel=0.7999999999999999, colsample_bynode=1,
       colsample_bytree=0.7, gamma=0, gpu_id=-1, importance_type='gain',
       interaction_constraints='', learning_rate=0.11, max_delta_step=0,
       max_depth=5, min_child_weight=1, missing=nan,
       monotone_constraints='()', n_estimators=350, n_jobs=6,
       num_parallel_tree=1, objective='reg:squarederror', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
       tree_method='exact', validate_parameters=1, verbosity=None)
1random_best_param : {'n_estimators': 350, 'max_depth': 5, 'learning_rate': 0.11, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7999999999999999}

2random_best_estimator : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
       colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.11, max_delta_step=0, max_depth=4,
       min_child_weight=1, missing=nan, monotone_constraints='()',
       n_estimators=390, n_jobs=6, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)
2random_best_param : {'n_estimators': 390, 'max_depth': 4, 'learning_rate': 0.11, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6}

3random_best_estimator : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
       colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.11, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=nan, monotone_constraints='()',
       n_estimators=310, n_jobs=6, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)
3random_best_param : {'n_estimators': 310, 'max_depth': 6, 'learning_rate': 0.11, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7}

'''
# ----------------------------------------------------------------------------------------------------------------------------------------------------
'''
0random_best_param : {'n_estimators': 350, 'max_depth': 6, 'learning_rate': 0.060000000000000005, 'colsample_bytree': 0.8999999999999999, 'colsample_bylevel': 0.7}   
1random_best_param : {'n_estimators': 350, 'max_depth': 5, 'learning_rate': 0.11, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7999999999999999}
2random_best_param : {'n_estimators': 390, 'max_depth': 4, 'learning_rate': 0.11, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6}
3random_best_param : {'n_estimators': 310, 'max_depth': 6, 'learning_rate': 0.11, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7}
'''

'''
{'n_estimators': [310, 350, 390], 'max_depth': [4, 5, 6], 'learning_rate': [0.06, 0.11], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9], 'colsample_bylevel': [0.6, 0.7, 0.8]}
'''