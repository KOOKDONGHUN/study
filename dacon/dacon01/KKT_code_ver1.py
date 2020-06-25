import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
# test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

# x_train = np.load('./dacon/comp1/x_train.npy')
# y_train = np.load('./dacon/comp1/y_train.npy')
# x_pred = np.load('./dacon/comp1/x_pred.npy')


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

# 2. model
final_y_pred = []
parameter = [{'colsample_bytree':list(np.arange(0.6,0.9,0.1)),
              'max_depth': [4,5,6],
              'n_estimators': list(np.arange(150,400,10)),
              'learning_rate': list(np.arange(0.01,0.56,0.05)),
              'colsample_bylevel': list(np.arange(0.6,0.9,0.1))}
]

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

print(type(x_train))
print(type(y_train))
print(type(x_pred))

# 모델 컬럼별 4번
for i in range(4):
    model = XGBRegressor()
    model.fit(x_train,y_train[:,i])
    score = model.score(x_test,y_test[:,i])
    print("r2 : ", score)
    thresholds = np.sort(model.feature_importances_)
    print(thresholds)
    best_score = score
    best_model = model
    best_y_pred = model.predict(x_pred)
    print(best_y_pred.shape)
    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 이 둘중 하나 쓰는거 이해하면 사용 가능
                                                ## 이거 주어준 값 이하의 중요도를 가진 feature를 전부 자르는 파라미터
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)

        print(select_x_train.shape)

        selection_model = GridSearchCV(XGBRegressor(), parameter, n_jobs=-1, cv = 5)
        selection_model.fit(select_x_train, y_train[:,i])

        y_pred = selection_model.predict(select_x_test)
        score = r2_score(y_test[:,i],y_pred)
        print(selection_model.best_params_)
        if score >= best_score:
            best_score = score
            best_model = selection_model
            best_y_pred = selection_model.predict(select_x_pred)
        print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    final_y_pred.append(best_y_pred)


final_y_pred = np.array(final_y_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": final_y_pred[0,:],
    "hbo2": final_y_pred[1,:],
    "ca": final_y_pred[2,:],
    "na": final_y_pred[3,:]
})

submissions.to_csv('./_sub_.csv', index = False)