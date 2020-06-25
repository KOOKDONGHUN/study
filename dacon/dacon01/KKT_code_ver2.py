import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./data/dacon/comp1/x_train.npy')
y_train = np.load('./data/dacon/comp1/y_train.npy')
x_pred = np.load('./data/dacon/comp1/x_pred.npy')

# print(x_train.shape)
# print(x_pred.shape)

# x_train2 = pd.read_csv('./data/dacon/comp1/new_test_nan_interpolate.csv')
# x_pred2 = pd.read_csv('./data/dacon/comp1/new_train_nan_interpolate.csv')

# x_train2 = x_train2.drop('id',axis=1)
# x_pred2 = x_pred2.drop('id',axis=1)

# x_train2 = x_train2.loc[:, 'rho':'990_src'].values
# x_pred2 = x_pred2.loc[:, 'rho':'990_src'].values

# print(x_train2)
# print(x_pred2)

# print(x_train)
# print(x_pred)

# x_train = x_train[:, -140 :]
# x_pred = x_pred[:, -140 :]

# print(x_train.shape)
# print(x_pred.shape)

# print(x_train2.shape)
# print(x_pred2.shape)

# x_train = np.concatenate((x_train2,x_train), axis=1)
# print(x_train)
# print(x_train.shape)

# x_pred = np.concatenate((x_pred2,x_pred), axis=1)
# print(x_pred)
# print(x_pred.shape)


# print(x_train.shape)
# print(x_pred.shape)
""" 
ls = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 19, 21, 23, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
for i in range(71,141):
    ls.append(i)
print(len(ls))
tmp = list()
tmp2 = list()

for i in ls:
    tmp.append(x_train[:, i])
    tmp2.append(x_pred[:, i])

x_train = np.array(tmp)
x_pred = np.array(tmp2)

print(x_train.shape)
print(x_pred.shape)

x_train = x_train.transpose()
x_pred = x_pred.transpose() """

# print(x_train.shape)
# print(x_pred.shape)
# print(y_train.shape)

x_train = np.delete(x_train, (10, 11, 12, 13), axis=1)
x_pred = np.delete(x_pred, (10, 11, 12, 13), axis=1)


print(x_train.shape)
print(x_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)



# 2. model
parameters1 = {'n_estimators': [420],
    'learning_rate': [0.09],
    'colsample_bytree': [0.79],
    'max_depth': [5]
    }

parameters2 = {'n_estimators': [420],
    'learning_rate': [0.09],
    'colsample_bytree': [0.7999],
    'max_depth': [5]
    }

parameters3 = {'n_estimators': [418],
    'learning_rate': [0.12],
    'colsample_bytree': [0.7999],
    'max_depth': [7]
    }

parameters4 = {'n_estimators': [455],
    'learning_rate': [0.05],
    'colsample_bytree': [0.7],
    'max_depth': [9]
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
    "id": test.index,
    "hhb": y_pred[0,:],
    "hbo2": y_pred[1,:],
    "ca": y_pred[2,:],
    "na": y_pred[3,:]
})

submissions.to_csv('./data/dacon/comp1/submission_____.csv', index = False)