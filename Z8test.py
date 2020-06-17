import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 2. 월별 서울 온도 데이터 로드
temp_data_month = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_month.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False,index_col=0)
temp_data_month = temp_data_month.drop('지점',axis=1)

# 결측지 발견 20년6월의 평균온도 -> 일별데이터에서 현재까지의 평균을 이용하여 결측치를 채워보기
view_nan(temp_data_month,0)
index = temp_data_month.loc[pd.isna(temp_data_month[temp_data_month.columns[-1]]), :].index
# print(index)

# 3. 일별 서울 온도 데이터 로드
temp_data_day = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False,index_col=0)
temp_data_day = temp_data_day.drop('지점',axis=1)
# print(temp_data_day)

# 일별 온도데이터의 nan값을 찾고 그 행을 출력
view_nan(temp_data_day,0)
# print(temp_data_day.columns[-1])
index = temp_data_day.loc[pd.isna(temp_data_day[temp_data_day.columns[-1]]), :].index
# print(index) # 2017-10-12 의 최고 기온

# 일단 원하는건 현재까지의 6월의 월별 평균기온을 구하기위한 일별 평균기온
want = temp_data_day.loc['2020-06-01':,'평균기온(℃)'].values
# print(round(want.mean(),1))

# 월별 평균온도 결측치에 값을 대입 
temp_data_month = temp_data_month.fillna(round(want.mean(),1))
view_nan(temp_data_month,0)
