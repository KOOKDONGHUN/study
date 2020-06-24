import pandas as pd
import numpy as np
from hamsu import view_nan
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae

# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

y = train.loc[:, 'hhb' : 'na']
print(y)

train = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

# view_nan(train)
# print(train)
train = train.interpolate(axis=1)
# view_nan(train)
# print(train)
train = train.replace(0.0, 10)
# print(train)

test = test.interpolate(axis=1)
test = test.replace(0.0, 10)

# print(test)

train = np.log10(train)
# print(train)
test = np.log10(test)
# x_pred = np.log10(test)

# 0~35 src?
KKT_x = np.load('./data/dacon/comp1/x_train.npy')
KKT_pred = np.load('./data/dacon/comp1/x_pred.npy')
# print(KKT_x.shape)

# KKT_x = KKT_x[:, 35:]
# KKT_pred = KKT_pred[:, 35:]

test = test.values
train = train.values
# print(train)

train = np.concatenate((train,KKT_x),axis=1)
# print(train.shape)

x_pred = np.concatenate((test,KKT_pred),axis=1)
# print(x_pred.shape)

y = y.values

x_train, x_test, y_train, y_test = train_test_split(
    train, y, train_size = 0.8, random_state = 66
)



# 2. model
parameters1 = {'n_estimators': [450],
    'learning_rate': [0.08],
    'colsample_bytree': [0.79],
    'max_depth': [5]
    }

parameters2 = {'n_estimators': [450],
    'learning_rate': [0.08],
    'colsample_bytree': [0.88],
    'max_depth': [5]
    }

parameters3 = {'n_estimators': [450],
    'learning_rate': [0.09],
    'colsample_bytree': [0.88],
    'max_depth': [5]
    }

parameters4 = {'n_estimators': [450],
    'learning_rate': [0.1],
    'colsample_bytree': [0.88],
    'max_depth': [5]
    }


kfold1 = KFold(n_splits=5, shuffle=True, random_state=66)
kfold2 = KFold(n_splits=5, shuffle=True, random_state=66)
kfold3 = KFold(n_splits=5, shuffle=True, random_state=66)
# kfold4 = KFold(n_splits=4, shuffle=True, random_state=66)
kfold4 = KFold(n_splits=5, shuffle=True, random_state=66)

y_test_pred = []
y_pred = []

search1 = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameters1, cv = kfold1, n_iter=1)
search2 = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameters2, cv = kfold2, n_iter=1)
search3 = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameters3, cv = kfold3, n_iter=1)
search4 = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameters4, cv = kfold4, n_iter=1)
####################################0
fit_params = {
    'verbose': False,
    'eval_metric': ['logloss','mae'],
    'eval_set' : [(x_train,y_train[:,0]),(x_test,y_test[:,0])],
    'early_stopping_rounds' : 5
}
search1.fit(x_train, y_train[:,0],**fit_params)
y_pred.append(search1.predict(x_pred))
print(f'y_pred : {y_pred}')
y_test_pred.append(search1.predict(x_test))
print(f'y_test_pred : {y_test_pred}')
print(search1.best_score_)
#####################################1
fit_params = {
    'verbose': False,
    'eval_metric': ['logloss','mae'],
    'eval_set' : [(x_train,y_train[:,1]),(x_test,y_test[:,1])],
    'early_stopping_rounds' : 5
}
search2.fit(x_train, y_train[:,1],**fit_params)
y_pred.append(search2.predict(x_pred))
print(f'y_pred : {y_pred}')
y_test_pred.append(search2.predict(x_test))
print(f'y_test_pred : {y_test_pred}')
print(search2.best_score_)
#####################################2
fit_params = {
    'verbose': False,
    'eval_metric': ['logloss','mae'],
    'eval_set' : [(x_train,y_train[:,2]),(x_test,y_test[:,2])],
    'early_stopping_rounds' : 5
}
search3.fit(x_train, y_train[:,2],**fit_params)
y_pred.append(search3.predict(x_pred))
print(f'y_pred : {y_pred}')
y_test_pred.append(search3.predict(x_test))
print(f'y_test_pred : {y_test_pred}')
print(search3.best_score_)
#####################################3
fit_params = {
    'verbose': False,
    'eval_metric': ['logloss','mae'],
    'eval_set' : [(x_train,y_train[:,3]),(x_test,y_test[:,3])],
    'early_stopping_rounds' : 5
}
search4.fit(x_train, y_train[:,3],**fit_params)
y_pred.append(search4.predict(x_pred))
print(f'y_pred : {y_pred}')
y_test_pred.append(search4.predict(x_test))
print(f'y_test_pred : {y_test_pred}')
print(search4.best_score_)



y_pred = np.array(y_pred)
y_test_pred = np.array(y_test_pred)

# print(y_pred.shape)
# r2 = r2_score(y_test,y_test_pred)
# mae = mae(y_test,y_test_pred)
# print('r2 :', r2)
# print('mae :', mae)

# print(test.index)
# print(y_pred[0,:].shape)
# print(y_pred[1,:].shape)
# print(y_pred[2,:].shape)
# print(y_pred[3, :].shape)

submissions = pd.DataFrame({
    "id": np.arange(10000,20000),
    "hhb": y_pred[0,:],
    "hbo2": y_pred[1,:],
    "ca": y_pred[2,:],
    "na": y_pred[3,:]
})

submissions.to_csv('./data/dacon/comp1/submission_____.csv', index = False)