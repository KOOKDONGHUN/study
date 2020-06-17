''' RF1에서 말한대로 봄,여름, 가을 과 겨울을 분리하여 2개의 모델을 만들어 보자 '''

import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

'''시간별로 20개가 안되면 통계가 안잡함'''
# ----------------------------------------------------------------------------------------------------------------------
# 1-0. 데이터 x, y 구분하기

# 1-1. 데이터가 내가 원하는 형식으로 되어있지 않아서 제외
# cloud_data = pd.read_csv('./data/csv/Seoul/Seoul_cloud_2010-2020_by_month.csv',encoding='CP949') 형식이 이상해서 이렇게 하면 에러가남 밑에 줄 처럼 해야함
# cloud_data = pd.read_csv('./data/csv/Seoul/Seoul_cloud_2010-2020_by_month.csv',encoding='CP949',header=None,sep=',',error_bad_lines=False)

# 1-2. 월별 서울 온도 데이터 로드
temp_data_month = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_month.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)#,index_col=0)
temp_data_month = temp_data_month.drop('지점',axis=1)
# print(temp_data_month)

temp_data_month.columns = ['시간', '평균기온', '최저기온', '최고기온']
# print(temp_data_month)
# 결측지 발견 20년6월의 평균온도 -> 일별데이터에서 현재까지의 평균을 이용하여 결측치를 채워보기
view_nan(temp_data_month,0)
index = temp_data_month.loc[pd.isna(temp_data_month[temp_data_month.columns[-1]]), :].index
# print(index)


# 1-3. 일별 서울 온도 데이터 로드
temp_data_day = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)#,index_col=0)
temp_data_day = temp_data_day.drop('지점',axis=1)
# print(temp_data_day)

temp_data_day.columns = ['시간', '평균기온', '최저기온', '최고기온']
# print(temp_data_day)

# 일별 온도데이터의 nan값을 찾고 그 행을 출력
view_nan(temp_data_day,0) # 최고 기온에 결측치 존재
# print(temp_data_day.columns[-1])
index = temp_data_day.loc[pd.isna(temp_data_day[temp_data_day.columns[-1]]), :].index
# print(index) # 2017-10-12 의 최고 기온


# 1-4. 월별 평균온도 결측치에 값을 대입 
# 일단 원하는건 현재까지의 6월의 월별 평균기온을 구하기위한 일별 평균기온
mean = round(temp_data_day.iloc[-11:, 1].mean(),1)
temp_data_month = temp_data_month.fillna(mean)
view_nan(temp_data_month,0)

data = temp_data_month.values
# print(data.shape)
# print(len(data))


# 1-5. 일별 미세먼지 데이터 로드 월별 데이터에 6월의 평균이 없음 이걸로 일단 구해서 채워보자
dust_data_day = pd.read_csv('./data/Seoul/Seoul_dust_2010-2020_by_day.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)#,index_col=2)
dust_data_day = dust_data_day.drop('지점번호',axis=1)
dust_data_day = dust_data_day.drop('지점명',axis=1)

dust_data_day.columns = ['시간', '미세먼지농도']

# 보간법
dust_data_day = dust_data_day.interpolate()
view_nan(dust_data_day)
# print(dust_data_day)

# 월별 결측치 6월을 채우기위한 11일까지의 평균 미세먼지 농도
dust_mont_6_mean = round(dust_data_day.iloc[-11:,1].values.mean(),1)
# print(dust_mont_6_mean)


# 1-6. 월별 미세먼지 데이터 로드
dust_data_month = pd.read_csv('./data/Seoul/Seoul_dust_2010-2020_by_month.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)#,index_col=2)
dust_data_month = dust_data_month.drop('지점번호',axis=1)
dust_data_month = dust_data_month.drop('지점명',axis=1)

dust_data_month.columns = ['시간', '미세먼지농도']

# print(dust_data_month)

view_nan(dust_data_month,0)

index = dust_data_month.loc[pd.isna(dust_data_month[dust_data_month.columns[-1]]), :].index
# print(index) # 42

# print(dust_data_month.iloc[42])

dust_data_month = dust_data_month.interpolate()

view_nan(dust_data_month)

# print(dust_data_month)

series_raw = pd.Series(data=['2020-06', dust_mont_6_mean],index=['시간','미세먼지농도'])
dust_data_month = dust_data_month.append(series_raw,ignore_index=True)

# print(dust_data_month)


# 1-7. 강수량 월별 데이터 로드
rain_data_month = pd.read_csv('./data/Seoul/Seoul_rain_2010-2020_by_month.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)#,index_col=2)
# print(rain_data_month)
rain_data_month = rain_data_month.drop('지점',axis=1)
# print(rain_data_month)
rain_data_month.columns = ['시간', '강수량']
# print(rain_data_month)
view_nan(rain_data_month)



data = pd.merge(temp_data_month,dust_data_month,on='시간')
data = pd.merge(data,rain_data_month,on='시간')
# print(data)
view_nan(data)

data_origin = data.drop('시간',axis=1)
# print(data_origin)
data = data_origin.values


# ----------------------------------------------------------------------------------------------------------------------
# 2-1. 데이터 x, y 구분하기 (봄, 여름, 가을 과 겨울)
x_data = []
y_data = []

for i in range(len(data)-1):
    x_data.append(data[i,:])
    y_data.append(data[i+1,:3])

x_data = np.array(x_data)
y_data = np.array(y_data)

# print(x_data.shape)
# print(y_data.shape)

x_train,x_test,y_train, y_test = train_test_split(x_data,y_data,shuffle=False,test_size=0.1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model = RandomForestRegressor()
model.fit(x_train,y_train)
# res = model.score(x_test,y_test)
y_pred = model.predict(x_test)




for i in range(len(y_pred)):
    if 6+i <13:
        print(f"2019-{6+i}\t y_tets : {y_test[i]} \t y_pred : {y_pred[i].round(1)}")
    else:
        print(f"2020-{i-6}\t y_tets : {y_test[i]} \t y_pred : {y_pred[i].round(1)}")


r2_y_pred = r2_score(y_test,y_pred)
print("r2 : ",r2_y_pred)

np.save
x_pred = np.load('./Data/Seoul/5col_x_pred.npy')
print(x_pred)
y_past = np.load('./Data/Seoul/5col_y_past.npy')
y_pred = model.predict(x_pred)
print(f"RF -> 실제값 : {y_past} \t 예측값 : {y_pred}")


data_origin.columns = ['avg temp', 'low temp', 'high temp', 'dust','rain']
plot_feature_importances(model,data_origin)