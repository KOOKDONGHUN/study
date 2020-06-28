from xgboost import XGBClassifier, plot_importance, XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score,GridSearchCV
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel

train = np.load('./data/dacon/comp1/train.npy')
test = np.load('./data/dacon/comp1/pred.npy')

x = train[:, :-4]
y = train[:, -4:]

print(x)
print(y)



# 회기 모델
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

                                                    
# n_estimators = 450
# learning_rate = 0.1
# colsample_bytree = 0.85
# colsample_bylevel = 0.9

# max_depth = 6
# n_jobs = 6

parameters = [{"n_estimators": [2000],
              "learning_rate": [0.01],
              "max_depth": [5],
              "colsample_bytree": [0.79],
              "colsample_bylevel": [0.79]}]

parameters2 = [{"n_estimators": [2000],
              "learning_rate": [0.01],
              "max_depth": [6],
              "colsample_bytree": [0.79],
              "colsample_bylevel": [0.79]}]

kfold = KFold(n_splits=4, shuffle=True, random_state=66)

model = XGBRegressor(n_jobs=6)
model2 = XGBRegressor(n_jobs=6)

name_ls = ['hhb','hbo2','ca','na']
tmp_dic = dict()

###
model.fit(x_train,y_train[:, 2])
thresholds_2 = np.sort(model.feature_importances_)

model.fit(x_train,y_train[:, 3])
thresholds_3 = np.sort(model.feature_importances_)

###
selection_2=SelectFromModel(model,threshold=thresholds_2[100],prefit=True)
selection_3=SelectFromModel(model,threshold=thresholds_3[100],prefit=True)

selection_x_train_2 = selection_2.transform(x_train)
selection_x_train_3 = selection_3.transform(x_train)

###
selection_x_test_2 = selection_2.transform(x_test)
selection_x_test_3 = selection_3.transform(x_test)

###
test_2 = selection_2.transform(test)
test_3 = selection_3.transform(test)

###
model = GridSearchCV(model, parameters, cv = kfold)
model2 = GridSearchCV(model2, parameters2, cv = kfold)


## hbb, hbo2
for i in range(2):
    model.fit(x_train,y_train[:, i])

    y_test_pred = model.predict(x_test)
    r2 = r2_score(y_test[:, i],y_test_pred)
    print(f"r2 : {r2}")
    mae = mean_absolute_error(y_test[:, i],y_test_pred)
    print(f"mae : {mae}")
    
    y_pred = model.predict(test)
    tmp_dic[name_ls[i]] = y_pred

## ca
model2.fit(selection_x_train_2,y_train[:, 2])

y_test_pred = model2.predict(selection_x_test_2)
r2 = r2_score(y_test[:, 2],y_test_pred)
print(f"r2 : {r2}")
mae = mean_absolute_error(y_test[:, 2],y_test_pred)
print(f"mae : {mae}")

y_pred = model2.predict(test_2)
tmp_dic[name_ls[i]] = y_pred

## na
model2.fit(selection_x_train_3,y_train[:, 3])

y_test_pred = model2.predict(selection_x_test_3)
r2 = r2_score(y_test[:, 3],y_test_pred)
print(f"r2 : {r2}")
mae = mean_absolute_error(y_test[:, 3],y_test_pred)
print(f"mae : {mae}")

y_pred = model2.predict(test_3)
tmp_dic[name_ls[i]] = y_pred


df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])

# print(df)

df.to_csv('./submission.csv',index_label='id')