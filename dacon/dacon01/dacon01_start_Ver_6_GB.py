import pandas as pd                      
import numpy as np                         
import matplotlib.pyplot as plt            
import seaborn as sns                      

import xgboost as xgb                       
from sklearn.model_selection import KFold   

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings(action='ignore') 

train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

test = test.fillna(train.mean())
train = train.fillna(train.mean())

x_train = train.loc[:, '650_dst':'990_dst'].values
test = test.loc[:, '650_dst':'990_dst'].values
y_train = train.loc[:, 'hhb':'na'].values

model = GradientBoostingRegressor(max_depth=3) 


print(x_train.shape, y_train.shape)

def model_fit(y,col) :
    model.fit(x_train,y)
    data = model.predict(test)
    df = pd.Series(data)
    return df
print(y_train[:,0].shape)

hhb = model_fit(y_train[:, 0],'hbb')
hbo2 = model_fit(y_train[:, 1],'hbo2')
ca = model_fit(y_train[:, 2],'ca')
na = model_fit(y_train[:, 3],'na')

df = pd.DataFrame([hhb,hbo2,ca,na])

df = df.transpose()

print(df.head(10))

# indexss = 
# print(indexss)

df.index =[i for i in range(10000,20000,1)]

df.to_csv('./data/dacon/comp1/sample_submission_GB.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

print(df)

# 핏팅에서 target값을 1개씩 줘야함 
# model.fit(x_train,y_train)

print(model.feature_importances_)
