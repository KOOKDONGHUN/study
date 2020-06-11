import pandas as pd                      
import numpy as np                         
import matplotlib.pyplot as plt            
import seaborn as sns                      

import xgboost as xgb                       
from sklearn.model_selection import KFold   

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings(action='ignore') 

train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

test = test.fillna(train.mean())
train = train.fillna(train.mean())

x_train = train.loc[:, '650_dst':'990_dst']
y_train = train.loc[:, 'hhb':'na']

print(x_train.shape, y_train.shape)

model = RandomForestRegressor() 

model.fit(x_train,y_train)

print(model.feature_importances_)

submission.to_csv('./data/dacon/comp1/Dacon_baseline.csv', index=False)