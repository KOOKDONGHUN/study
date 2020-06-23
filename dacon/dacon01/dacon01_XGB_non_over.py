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

# data
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')


# missing value
train = train.transpose()
test = test.transpose()

train = train.interpolate()
test = test.interpolate()

train = train.transpose()
test = test.transpose()

train = train.drop('id',axis=1)
test = test.drop('id',axis=1)

# print(train)
# print(test)

# filtering over values
def outliners(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    upper_bound = q3+iqr*1.5
    lower_bound = q1-iqr*1.5
    return np.where((data>upper_bound) | (data<lower_bound))

tmp_ls = list()

for i in range(len(train.columns)):
    # print("-idx-",i)

    x=outliners(train.iloc[:,i])
    # print("-x-",x)
    
    l = len(list(x[0]))
    # print("-len(x)-", l)

    if l == 0 :
        tmp_ls.append(i)


# split x, y
# x = train.loc[:, 'rho':'990_dst']
# test = test.loc[:, 'rho':'990_dst']

x = train.iloc[:, tmp_ls]
test = test.iloc[:, tmp_ls]

print(x)
print(test)

y = train.loc[:, 'hhb':'na']

# split train, test
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,
                                                    random_state=0)


model1 = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
       colsample_bynode=1, colsample_bytree=0.7999999999999999, gamma=0,
       gpu_id=-1, importance_type='gain', interaction_constraints='',
       learning_rate=0.11, max_delta_step=0, max_depth=6,
       min_child_weight=1,  monotone_constraints='()',
       n_estimators=390, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)
model2 = XGBRegressor(base_score=0.5, booster='gbtree',
       colsample_bylevel=0.8999999999999999, colsample_bynode=1,
       colsample_bytree=0.7999999999999999, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.11, max_delta_step=0, max_depth=6,
       min_child_weight=1,  monotone_constraints='()',
       n_estimators=390, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)
model3 = XGBRegressor(base_score=0.5, booster='gbtree',
       colsample_bylevel=0.7999999999999999, colsample_bynode=1,
       colsample_bytree=0.7999999999999999, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.060000000000000005, max_delta_step=0, max_depth=6,
       min_child_weight=1,  monotone_constraints='()',
       n_estimators=350, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)
model4 = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
       colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.060000000000000005, max_delta_step=0, max_depth=6,
       min_child_weight=1,  monotone_constraints='()',
       n_estimators=390, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
       validate_parameters=1, verbosity=None)


model1.fit(x_train,y_train.iloc[:, 0])
model2.fit(x_train,y_train.iloc[:, 1])
model3.fit(x_train,y_train.iloc[:, 2])
model4.fit(x_train,y_train.iloc[:, 3])

print(f'model1 == r2 : {r2_score(y_test.iloc[: ,0],model1.predict(x_test))} \t mae : {mean_absolute_error(y_test.iloc[: ,0],model1.predict(x_test))}')
print(f'model2 == r2 : {r2_score(y_test.iloc[: ,1],model2.predict(x_test))} \t mae : {mean_absolute_error(y_test.iloc[: ,1],model2.predict(x_test))}')
print(f'model3 == r2 : {r2_score(y_test.iloc[: ,2],model3.predict(x_test))} \t mae : {mean_absolute_error(y_test.iloc[: ,2],model3.predict(x_test))}')
print(f'model4 == r2 : {r2_score(y_test.iloc[: ,3],model4.predict(x_test))} \t mae : {mean_absolute_error(y_test.iloc[: ,3],model4.predict(x_test))}')

y_pred1 = model1.predict(test)
y_pred2 = model2.predict(test)
y_pred3 = model3.predict(test)
y_pred4 = model4.predict(test)

name_ls = ['hhb','hbo2','ca','na']

tmp_dict = dict()

tmp_dict[name_ls[0]] = y_pred1
tmp_dict[name_ls[1]] = y_pred2
tmp_dict[name_ls[2]] = y_pred3
tmp_dict[name_ls[3]] = y_pred4

df = pd.DataFrame(tmp_dict,range(10000,20000),columns=['hhb','hbo2','ca','na'])
df.to_csv('./submission__.csv',index_label='id')