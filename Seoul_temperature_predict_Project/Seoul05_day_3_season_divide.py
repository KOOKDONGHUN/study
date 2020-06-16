import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense,LSTM,Dropout
# from keras.callbacks import EarlyStopping

''' 일평균 온도, 최저온도, 최고온도, 미세먼지농도, 강수량'''

# 1-0. 학습과 테스트 데이터 분리 및 스케일링
data = pd.read_csv('./data/Seoul/Seoul_data.csv',index_col=0)
print(data)

print(data.index[0][5:7])
print(type(data.index[0][5:7]))

tmp = list()
'''봄 0 여름 1 가을 2 겨울 3'''
for i in data.index:
    if i[5:7] == '01':
        tmp.append(3)
    elif i[5:7] == '02':
        tmp.append(3)
    elif i[5:7] == '03':
        tmp.append(0)
    elif i[5:7] == '04':
        tmp.append(0)
    elif i[5:7] == '05':
        tmp.append(0)
    elif i[5:7] == '06':
        tmp.append(1)
    elif i[5:7] == '07':
        tmp.append(1)
    elif i[5:7] == '08':
        tmp.append(1)
    elif i[5:7] == '09':
        tmp.append(2)
    elif i[5:7] == '10':
        tmp.append(2)
    elif i[5:7] == '11':
        tmp.append(2)
    elif i[5:7] == '12':
        tmp.append(3)
    else : 
        print("뭔가 잘못 됬다")
data['season'] = tmp
print(data)
# price_df["month"] =[i[-5:-3] for i in list(price_df.index)]
# price_df["year"] =[i[:4] for i in list(price_df.index)]

spring = data[data['season']==0]
print(spring)

summer = data[data['season']==1]
print(summer)

fall = data[data['season'] == 2]
print(fall)

winter = data[data['season'] == 3]
print(winter)

spring.to_csv('./data/Seoul/Seoul_spring.csv')
summer.to_csv('./data/Seoul/Seoul_summer.csv')
fall.to_csv('./data/Seoul/Seoul_fall.csv')
winter.to_csv('./data/Seoul/Seoul_winter.csv')