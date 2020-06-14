import pandas as pd
import numpy as np
import csv
from hamsu import view_nan

# 데이터가 내가 원하는 형식으로 되어있지 않아서 제외
# cloud_data = pd.read_csv('./data/csv/Seoul/Seoul_cloud_2010-2020_by_month.csv',encoding='CP949') 형식이 이상해서 이렇게 하면 에러가남 밑에 줄 처럼 해야함
# cloud_data = pd.read_csv('./data/csv/Seoul/Seoul_cloud_2010-2020_by_month.csv',encoding='CP949',header=None,sep=',',error_bad_lines=False)

temp_data_month = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_month.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
# print(temp_data_month)
view_nan(temp_data_month,0)

temp_data_day = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
# print(temp_data_day)

# 날짜별 온도데이터의 nan값을 찾고 그 행을 출력
view_nan(temp_data_day,0)
# print(temp_data_day.columns[-1])

index = temp_data_day.loc[pd.isna(temp_data_day[temp_data_day.columns[-1]]), :].index
# print(index)
# print(temp_data_day.iloc[index,])

# temp_data_month = temp_data_month.fillna()