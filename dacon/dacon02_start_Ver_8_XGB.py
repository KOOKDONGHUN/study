from platform import python_version
import pandas as pd
import numpy as np

# 입력하세요.
import sklearn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb                       # XGBoost 패키지
from sklearn.model_selection import KFold 

import warnings
warnings.filterwarnings(action='ignore') 

# 1. data
train_features = pd.read_csv('./data/dacon/comp2/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv', index_col='id')
test_features = pd.read_csv('./data/dacon/comp2/test_features.csv')
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv')

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
def train_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]
    
        d_train = xgb.DMatrix(data = x_train, label = y_train)
        d_val = xgb.DMatrix(data = x_val, label = y_val)
        
        wlist = [(d_train, 'train'), (d_val, 'eval')]
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed':777
            }

        model = xgb.train(params=params, dtrain=d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
        print(f"model : {model}")
        models.append(model)
    
    return models

models = {}
for label in train_target.columns:
    # print('train column : ', label)
    models[label] = train_model(train_features, train_target[label])

for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(test_features.loc[:, :])))
    pred = np.mean(preds, axis=0)

    submission[col] = pred

submission.to_csv('./data/dacon/comp2/submission_XGB.csv', index=False)