import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1-0. 데이터 로드 및 결측치 제거
temperature = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
temperature = temperature.drop('지점',axis=1)
temperature.columns = ['date', 'avg', 'low', 'high']
# print(temperature)

dust = pd.read_csv('./data/Seoul/Seoul_dust_2010-2020_by_day.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)
dust = dust.drop('지점번호',axis=1)
dust = dust.drop('지점명',axis=1)
dust.columns = ['date', 'dust']
# print(dust)

rain = pd.read_csv('./data/Seoul/Seoul_rain_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
rain = rain.drop('지점',axis=1)
rain.columns = ['date', 'rain']
# print(rain)

# print("온도의 결측 확인")
view_nan(temperature)
index = temperature.loc[pd.isna(temperature[temperature.columns[-1]]), :].index
# print(temperature.iloc[index,:])
temperature = temperature.interpolate()
view_nan(temperature)
# print("온도의 결측 확인\n")

# print("미세먼지 결측치 확인")
view_nan(dust)
index = dust.loc[pd.isna(dust[dust.columns[-1]]), :].index
# print(dust.iloc[index,:])
dust = dust.interpolate()
view_nan(dust)
# print("미세먼지 결측치 확인")

# print('강수량 결측치 확인')
view_nan(rain)
index = rain.loc[pd.isna(rain[rain.columns[-1]]), :].index
# print(rain.iloc[index,:])
rain = rain.fillna(0)
view_nan(rain)
index = rain.loc[pd.isna(rain[rain.columns[-1]]), :].index
# print(rain.iloc[index,:])
rain = rain.fillna(method='bfill')
view_nan(rain)
# print('강수량 결측치 확인')

data = pd.merge(temperature,dust,on='date')
data = pd.merge(data,rain,on='date')

# print(data)
view_nan(data)

# data = data.drop('date',axis=1)
data = data.round(1)
print(data)
data.to_csv('./data/Seoul/Seoul_data.csv',index=False)
np.save('./Data/Seoul/Seoul_data.npy',arr=data)