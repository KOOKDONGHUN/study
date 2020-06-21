from xgboost import XGBClassifier, plot_importance, XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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


# feature importance columns index setting
hhb_feature = [0, 51, 52, 50, 49, 53, 16, 54, 70, 14, 17, 68, 15, 48, 55]
hbo2_feature = [70, 0, 35, 1, 34, 49, 48, 50, 51, 69, 52, 41, 13, 40, 33]
ca_feature = [69, 68, 67, 0, 34, 1, 64, 70, 31, 33, 35, 66, 2, 30, 13]
na_feature = [62, 63, 1, 56, 60, 57, 58, 55, 44, 65, 27, 50, 29, 45, 61]

def plus_one(hhb_feature):
    for i in range(len(hhb_feature)):
        hhb_feature[i] += 1
    return hhb_feature

hhb_feature = plus_one(hhb_feature)
hbo2_feature = plus_one(hbo2_feature)
ca_feature = plus_one(ca_feature)
na_feature = plus_one(na_feature)


# fill missing value
train = train.interpolate()
train = train.dropna(axis=0)

test = test.interpolate()
test = test.fillna(0)


# x, y split
# x, y devide
def x_y_split(index, feature):
    x = train.iloc[:,feature]
    y = train.iloc[:,-4+index]
    return x, y

hhb_x, hhb_y = x_y_split(0,hhb_feature)
hbo2_x ,hbo2_y = x_y_split(1,hbo2_feature)
ca_x, ca_y = x_y_split(2,ca_feature)
na_x, na_y = x_y_split(3,na_feature)


# test feature importance divide
def pred_x_feature_importance(feature):
    x = test.iloc[:,feature]
    return x

x_pred_hhb = pred_x_feature_importance(hhb_feature)
x_pred_hbo2 = pred_x_feature_importance(hbo2_feature)
x_pred_ca = pred_x_feature_importance(ca_feature)
x_pred_na = pred_x_feature_importance(na_feature)



# train_test_plit
hhb_x, hhb_x_test, hhb_y, hhb_y_test = train_test_split(hhb_x,hhb_y,test_size=0.1,random_state=6)
hbo2_x, hbo2_x_test, hbo2_y, hbo2_y_test = train_test_split(hbo2_x,hbo2_y,test_size=0.1,random_state=6)
ca_x, ca_x_test, ca_y, ca_y_test = train_test_split(ca_x,ca_y,test_size=0.1,random_state=6)
na_x, na_x_test, na_y, na_y_test = train_test_split(na_x,na_y,test_size=0.1,random_state=6)


# scalling
scaler = RobustScaler()
# scaler = MinMaxScaler()

hhb_x = scaler.fit_transform(hhb_x)
hhb_x_test = scaler.transform(hhb_x_test)
x_pred_hhb = scaler.transform(x_pred_hhb)

hbo2_x = scaler.fit_transform(hbo2_x)
hbo2_x_test = scaler.transform(hbo2_x_test)
x_pred_hbo2 = scaler.transform(x_pred_hbo2)

ca_x = scaler.fit_transform(ca_x)
ca_x_test = scaler.transform(ca_x_test)
x_pred_ca = scaler.transform(x_pred_ca)

na_x = scaler.fit_transform(na_x)
na_x_test = scaler.transform(na_x_test)
x_pred_na = scaler.transform(x_pred_na)


# modelling
parameters = [{'colsample_bytree':list(np.arange(0.6,0.9,0.1)),
              'max_depth': [4,5,6],
              'n_estimators': list(np.arange(150,400,20)),
              'learning_rate': list(np.arange(0.1,0.5,0.1)),
              'colsample_bylevel': list(np.arange(0.6,0.9,0.1))}
]

n_jobs = -1

model = XGBRegressor()
model = GridSearchCV(model, parameters,n_jobs=n_jobs, cv=3)

# fitting
name_ls = ['hhb','hbo2','ca','na']
tmp_dic = dict()
# train_x_ls = [f'{hhb_x}',f'{hbo2_x}',f'{ca_x}',f'{na_x}']
# train_y_ls = [f'{hhb_y}',f'{hbo2_y}',f'{ca_y}',f'{na_y}']
# pred_x_ls = [f'{x_pred_hhb}',f'{x_pred_hbo2}',f'{x_pred_ca}',f'{x_pred_na}']

def fit_pred(x_train, y_train, x_test, y_test,x_pred,i):

    model.fit(x_train,y_train)

    # test pred  r2, MAE
    y_test_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_test_pred)
    print(f"r2 : {r2}")
    mae = mean_absolute_error(y_test,y_test_pred)
    print(f"mae : {mae}")

    # submit pred
    y_pred = model.predict(x_pred)

    # create submit DataFrame 
    tmp_dic[name_ls[i]] = y_pred
    # print(f"feature importance : {model.feature_importances_}")
    # plot_importance(model)

    # plt.show()

fit_pred(hhb_x,hhb_y,hhb_x_test,hhb_y_test,x_pred_hhb,0)
fit_pred(hbo2_x,hbo2_y,hbo2_x_test,hbo2_y_test,x_pred_hbo2,1)
fit_pred(ca_x,ca_y,ca_x_test,ca_y_test,x_pred_ca,2)
fit_pred(na_x,na_y,na_x_test,na_y_test,x_pred_na,3)

# submit
df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])
df.to_csv('./submission.csv',index_label='id')