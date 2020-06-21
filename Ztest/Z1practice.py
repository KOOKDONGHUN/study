import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn
import warnings

warnings.filterwarnings('ignore')

from xgboost import XGBClassifier, plot_importance, XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler
from hamsu import view_nan
import pandas as pd
import numpy as np

# 1. data
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

x = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

x = x.interpolate()
test = test.interpolate()

index = x.loc[pd.isna(x[x.columns[0]]), :].index

y = train.loc[:, 'hhb':'na']

x = x.fillna(0)

test = test.fillna(0)

# 회기 모델
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=0)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

test = scaler.transform(test)



# 2. modeling
allAlgorithms = all_estimators(type_filter='regressor')

max = 0

for (name,algorithm) in allAlgorithms:
    model = algorithm()
    try :
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        score = model.score(y_test, y_pred)
        if score >= max:
            max = score
            print(f"best model name {name}, best score {max}")
        print(name,"의 정답률 = ", score)
    except :
        print("error 1 !!",name)
        model = MultiOutputRegressor(algorithm())
        try :
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            score = model.score(y_test, y_pred)
            if score >= max:
                max = score
                print(f"best model name {name}, best score {max}")
            print(name,"의 정답률 = ", score)
        except :
            print("error 2 !!",name)

