import pandas as pd
import numpy as np





# 4. predict
# 1-0. 월별 서울 온도 데이터 로드
temp = pd.read_csv('./data/Seoul2/pred_temp.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
temp.columns = ['날짜', '지점','avg', 'low', 'high']
print(temp)

rain = pd.read_csv('./data/Seoul2/pred_rain.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
rain.columns = ['날짜', '지점','rain']
print(rain)

dust = pd.read_csv('./data/Seoul2/pred_dust.csv',encoding='CP949',header=3,sep=',',error_bad_lines=False)
dust.columns = ['지점', '지점명','날짜', 'dust']
print(dust)

data = pd.merge(temp,rain,on='날짜',how='outer')
data = pd.merge(data,dust,on='날짜',how='outer')
print(data)
data = data.drop(['지점_x','지점_y','지점', '지점명'],axis=1)
print(data)

data = data.fillna(0)

x_test = data.values

print(x_test.shape)