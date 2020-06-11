from platform import python_version
import pandas as pd
import numpy as np

# 입력하세요.
import sklearn
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import KFold 

import warnings
warnings.filterwarnings(action='ignore') 

# 1. data
train_features = pd.read_csv('./data/dacon/comp2/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv', index_col='id')
test_features = pd.read_csv('./data/dacon/comp2/test_features.csv')
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv')

print(train_target.head())

print(f"train_features {train_features.shape}")
print(f"train_target {train_target.shape}")
print(f"test_features {test_features.shape}")
'''train_features (1050000, 6)
train_target (2800, 4)
test_features (262500, 6)'''

def preprocessing_KAERI(data):

    # 충돌체 별로 0.000116초 까지의 가속도 데이터만 활용해보기
    _data = data.groupby('id').head(30)

    # string  형태로 변환
    _data['Time'] = _data["Time"].astype('str')

    # RandomForest 모델에 입력 할 수 있는 1차원 형태로 가속도 데이터 변환
    _data = _data.pivot_table(index='id',columns='Time', values = ['S1','S2','S3','S4'])

    # column 명 변환 왜?
    _data.columns = ['_'.join(col) for col in _data.columns.values]

    return _data

train_features = preprocessing_KAERI(train_features)
test_features = preprocessing_KAERI(test_features)

print(f"train_features {train_features.shape}")
print(f"test_features {test_features.shape}")


# 2. model
model = GradientBoostingRegressor() 

print(train_features.shape, train_target.shape)

def model_fit(y) :
    model.fit(train_features,y)
    data = model.predict(train_features)
    df = pd.Series(data)
    return df
print("train_target",train_target.shape)

X = model_fit(train_target[:, ])
Y = model_fit(train_target[:, 2])
M = model_fit(train_target[:, 3])
V = model_fit(train_target[:, 4])

df = pd.DataFrame([X,Y,M,V])

df = df.transpose()

print(df.head(10))

# df.to_csv('./data/dacon/comp1/sample_submission_GB.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')