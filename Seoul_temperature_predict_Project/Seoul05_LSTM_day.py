import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터 로드
temperature = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
temperature = temperature.drop('지점',axis=1)
temperature.columns = ['date', 'avg', 'low', 'high']
print(temperature)

dust = pd.read_csv('./data/Seoul/Seoul_dust_2010-2020_by_day.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)
dust = dust.drop('지점번호',axis=1)
dust = dust.drop('지점명',axis=1)
dust.columns = ['date', 'dust']
print(dust)

rain = pd.read_csv('./data/Seoul/Seoul_rain_2010-2020_by_day.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
rain = rain.drop('지점',axis=1)
rain.columns = ['date', 'rain']
print(rain)

print("온도의 결측 확인")
view_nan(temperature)
index = temperature.loc[pd.isna(temperature[temperature.columns[-1]]), :].index
print(temperature.iloc[index,:])
temperature = temperature.interpolate()
view_nan(temperature)
print("온도의 결측 확인\n")

print("미세먼지 결측치 확인")
view_nan(dust)
index = dust.loc[pd.isna(dust[dust.columns[-1]]), :].index
print(dust.iloc[index,:])
dust = dust.interpolate()
view_nan(dust)
print("미세먼지 결측치 확인")
view_nan(rain)


