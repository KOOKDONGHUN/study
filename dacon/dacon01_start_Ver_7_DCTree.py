import pandas as pd                      
import numpy as np                         
import matplotlib.pyplot as plt            
import seaborn as sns                      

import xgboost as xgb                       
from sklearn.model_selection import KFold   

from sklearn.tree import DecisionTreeRegressor
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

# pca1 = PCA(n_components=1)
# x_train = pca1.fit_transform(x_train)
# x_test = pca1.transform(x_test)

# y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1],)
model = DecisionTreeRegressor() # max_depth=3) # default? 몇이 좋냐고?

# encoder = LabelEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)

# print(x_train.shape, y_train.shape)

model.fit(x_train,y_train)

print(model.feature_importances_)

x_train_feature_name = ['650_dst','660_dst','670_dst','680_dst','690_dst','700_dst','710_dst','720_dst','730_dst','740_dst','750_dst','760_dst','770_dst','780_dst','790_dst','800_dst','810_dst','820_dst','830_dst','840_dst','850_dst','860_dst','870_dst','880_dst','890_dst','900_dst','910_dst','920_dst','930_dst','940_dst','950_dst','960_dst','970_dst','980_dst','990_dst']

def plot_feature_importances_cancer(model):
    n_features = x_train.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features),x_train_feature_name)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)

    plt.show()

plot_feature_importances_cancer(model)

df = pd.DataFrame(model.predict(test))

df.index =[i for i in range(10000,20000,1)]

df.to_csv('./data/dacon/comp1/sample_submission_DCTree.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')