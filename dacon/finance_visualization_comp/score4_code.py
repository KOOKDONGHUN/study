import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
import warnings 
warnings.filterwarnings('ignore')
color = sns.color_palette()

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import eli5
from eli5.sklearn import PermutationImportance

import shap

plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["font.family"] = 'NanumBarunGothic' # 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 폰트가 깨지는 것을 방지

train = pd.read_csv("./Data/credit_card_data.csv")
print(train.shape) # (3888, 26)

train['city'] = train['city'].fillna('전국')
train['sex'] = train['sex'].fillna('전체')
train['ages'] = train['ages'].apply(lambda x:int(x[:-1])).astype(float)
train['year_month'] = pd.to_datetime((train.year*100+train.month).apply(str),format='%Y%m')
train.drop(['year','month'], axis=1, inplace=True)
train = train[[train.columns[0],'year_month']+list(train.columns[1:-1])]

print(train.head(10))