from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold


from xgboost import XGBClassifier, plot_importance, XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler
from hamsu import view_nan
import pandas as pd
import numpy as np

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
test = test.fillna(0)


# 회기 모델
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# XGBRFRegressor??????

model = XGBRegressor()



model.fit(x_train,y_train)
model.booster().get_score()


thres_holds = np.sort(model.feature_importances_)
print(thres_holds)

model1 = MultiOutputRegressor(XGBRegressor())
model1.fit(x_train,y_train)

score = model.score(x_test,y_test)

print(f"r2 : {score}")

'''feature engineering'''




parameters = [{"n_estimators": [100, 200, 300],
              "learning_rate": [0.1, 0.3, 0.001, 0.01],
              "max_depth": [4, 5, 6]},
              
              {"n_estimators": [90, 100, 110],
              "learning_rate": [0.1, 0.001, 0.01],
              "max_depth": [3, 5, 7, 9],
              "colsample_bytree": [0.6, 0.9, 1]},

              {"n_estimators": [90, 100, 110],
              "learning_rate": [0.1, 0.001, 0.01],
              "max_depth": [3, 5, 7, 9],
              "colsample_bytree": [0.6, 0.9, 1],
              "colsample_bylevel": [0.6, 0.7, 0.9]}    ]
n_jobs = -1


# 반복문 안에다가 GridSearshCV를 엮어보기
for thresh in thres_holds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median

    selec_x_train = selection.transform(x_train)

    # print(f"selec_x_train.shape : {selec_x_train.shape}") # columns을 한개씩 줄이고 있다 

    # selec_model = XGBRegressor()
    selec_model = GridSearchCV(XGBRegressor(),parameters,cv=3, n_jobs=n_jobs)
    selec_model = MultiOutputRegressor(selec_model)
    selec_model.fit(selec_x_train,y_train)

    selec_x_test = selection.transform(x_test)
    y_pred = selec_model.predict(selec_x_test)

    score = r2_score(y_test,y_pred)
    # print(score)
    # print(f"model.feature_importances_ : {model.feature_importances_}")

    print(f"Thresh={np.round(thresh,2)} \t n={selec_x_train.shape[1]} \t r2={np.round(score*100,2)}")

# 메일 제목 : 아무개 **등