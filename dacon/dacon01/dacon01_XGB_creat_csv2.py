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

# x = train.loc[:, 'rho':'990_dst']
# test = test.loc[:, 'rho':'990_dst']

chk_cnt = 0

def outliners(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    upper_bound = q3+iqr*1.5
    lower_bound = q1-iqr*1.5
    return np.where((data>upper_bound) | (data<lower_bound))

for i in range(len(train.columns)):
    x=outliners(train.iloc[:,i])#열 별
    rows = list(x[0])#열 별로 값 가져옴
    for idx_col ,j in enumerate(rows):
        train.iloc[j,i]=np.nan

for i in range(len(test.columns)):
    x=outliners(test.iloc[:,i])#열 별
    rows = list(x[0])#열 별로 값 가져옴
    for idx_col ,j in enumerate(rows):
        test.iloc[j,i]=np.nan


print(train)
print(test)

train = train.transpose()
train = train.interpolate()
train = train.transpose()

test = test.transpose()
test = test.interpolate()
test = test.transpose()

train.to_csv('./data/dacon/comp1/new_train_nan_interpolate.csv', index= 0)
test.to_csv('./data/dacon/comp1/new_test_nan_interpolate.csv', index = 0)