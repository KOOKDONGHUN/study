import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1-0. 월별 서울 온도 데이터 로드
temp = pd.read_csv('./data/Seoul2/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
temp.columns = ['날짜', '지점','avg', 'low', 'high']
print(temp)

rain = pd.read_csv('./data/Seoul2/Seoul_rain_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
rain.columns = ['날짜', '지점','rain']
print(rain)

dust = pd.read_csv('./data/Seoul2/Seoul_dust_2010-2020_by_day.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)
dust.columns = ['지점', '지점명','날짜', 'dust']
print(dust)

data = pd.merge(temp,rain,on='날짜',how='outer')
data = pd.merge(data,dust,on='날짜',how='outer')
print(data)
data = data.drop(['지점_x','지점_y','지점', '지점명'],axis=1)
print(data)

# data.to_csv('./Data/Seoul2/merge_data.csv',index=False)