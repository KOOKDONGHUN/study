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

# 데이터
train = np.load('./data/dacon/comp1/train.npy')
test = np.load('./data/dacon/comp1/pred.npy')

x = train[:, :-4]
y = train[:, -4:]

# 회기 모델
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

test = scaler.transform(test)

n_estimators = 240
learning_rate = 0.1
colsample_bytree = 0.85
colsample_bylevel = 0.6

max_depth = 5
n_jobs = -1

model = XGBRegressor(  n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel,
                        max_depth=max_depth,
                        n_jobs=n_jobs   )

# model = MultiOutputRegressor(model)

name_ls = ['hhb','hbo2','ca','na']
tmp_dic = dict()

for i in range(len(y_train[0,:])):
    model.fit(x_train,y_train[:, i])

    y_test_pred = model.predict(x_test)
    r2 = r2_score(y_test[:, i],y_test_pred)
    print(f"r2 : {r2}")
    mae = mean_absolute_error(y_test[:, i],y_test_pred)
    print(f"mae : {mae}")
    
    y_pred = model.predict(test)
    tmp_dic[name_ls[i]] = y_pred

    print(f"feature importance : {model.feature_importances_}")
    plot_importance(model)

    plt.show()

df = pd.DataFrame(tmp_dic,range(10000,20000),columns=['hhb','hbo2','ca','na'])

print(df)

# df.to_csv('./submission.csv',index_label='id')