import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.utils import all_estimators
from sklearn.utils.testing import all_estimators
import sklearn
import warnings
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold



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
x_pred = test.fillna(0)


x =x.values
y = y.values
x_pred = x_pred.values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66
)

warnings.filterwarnings('ignore')


allAlgorithms = all_estimators(type_filter='regressor') # iris에 대한 모든 모델링

for index,(name, algorithm) in enumerate(allAlgorithms):
    model = algorithm() # -> 존나 희안한 문법인거 같은데 
    # try :
    model.fit(x_train,y_train[:, index])
    y_pred = model.predict(x_test)
    print(name,"의 정답률 = ", accuracy_score(y_test[:, index],y_pred))
    # except : 
    print("Error!!",name)

print(sklearn.__version__)