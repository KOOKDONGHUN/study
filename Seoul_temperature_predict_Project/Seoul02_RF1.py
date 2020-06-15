''' 전달과 다음달을 x와 y로 넣어서 데이터를 넣어봄
    데이터의 전처리를 하지 않고 단순히 머신러닝을 돌렸을때 r2가 0.8이 나옴 하지만 실제값과 실제 예측값을 비교해봤을때는 좋은 예측 결과는 아닌듯 보임
    대충 봤을때 영상의 날씨에 비해 상대적으로 영하나 0도에 가까울 수록 잘 맞추지 못하는 듯 보임
    봄,여름,가을 과 겨울을 분리하여 학습 시켜 보고싶다는 생각이 든다 '''

import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터가 내가 원하는 형식으로 되어있지 않아서 제외
# cloud_data = pd.read_csv('./data/csv/Seoul/Seoul_cloud_2010-2020_by_month.csv',encoding='CP949') 형식이 이상해서 이렇게 하면 에러가남 밑에 줄 처럼 해야함
# cloud_data = pd.read_csv('./data/csv/Seoul/Seoul_cloud_2010-2020_by_month.csv',encoding='CP949',header=None,sep=',',error_bad_lines=False)

# 2. 월별 서울 온도 데이터 로드
temp_data_month = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_month.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False,index_col=0)
temp_data_month = temp_data_month.drop('지점',axis=1)
# print(temp_data_month)
# 결측지 발견 20년6월의 평균온도 -> 일별데이터에서 현재까지의 평균을 이용하여 결측치를 채워보기
view_nan(temp_data_month,0)
index = temp_data_month.loc[pd.isna(temp_data_month[temp_data_month.columns[-1]]), :].index
# print(index)

# 3. 일별 서울 온도 데이터 로드
temp_data_day = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False,index_col=0)
temp_data_day = temp_data_day.drop('지점',axis=1)
# print(temp_data_day)
# 일별 온도데이터의 nan값을 찾고 그 행을 출력
view_nan(temp_data_day,0) # 최고 기온에 결측치 존재
# print(temp_data_day.columns[-1])
index = temp_data_day.loc[pd.isna(temp_data_day[temp_data_day.columns[-1]]), :].index
# print(index) # 2017-10-12 의 최고 기온 -> 보간법으로 채우자
# 일단 원하는건 현재까지의 6월의 월별 평균기온을 구하기위한 일별 평균기온
want = temp_data_day.loc['2020-06-01':,'평균기온(℃)'].values
# print(round(want.mean(),1))

# 월별 평균온도 결측치에 값을 대입 
temp_data_month = temp_data_month.fillna(round(want.mean(),1))
view_nan(temp_data_month,0)

data = temp_data_month.values
# print(data.shape)
# print(len(data))
# print(data[-1])

x_data = []
y_data = []

for i in range(len(data)-1):
    x_data.append(data[i,:])
    y_data.append(data[i+1,0])

x_data = np.array(x_data)
y_data = np.array(y_data)

# print(x_data.shape)
# print(y_data.shape)

x_train,x_test,y_train, y_test = train_test_split(x_data,y_data,shuffle=False,test_size=0.1)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = RandomForestRegressor()
model.fit(x_train,y_train)
# res = model.score(x_test,y_test)
y_pred = model.predict(x_test)

for i in range(len(y_pred)):
    if 6+i <13:
        print(f"2019-{6+i}\t y_tets : {y_test[i]} \t y_pred : {round(y_pred[i],1)}")
    else:
        print(f"2020-{i-6}\t y_tets : {y_test[i]} \t y_pred : {round(y_pred[i],1)}")
    # pass


r2_y_pred = r2_score(y_test,y_pred)
print("r2 : ",r2_y_pred)

''' 6월  예상 평균,       최저,      최고 
y_tets :     [23.2        14.8      32.8
y_pred :   [22.627      14.434    33.228     '''


temp_data_month.columns = ['avg temp', 'low temp', 'high temp']
plot_feature_importances(model,temp_data_month)