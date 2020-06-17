import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler

data = pd.read_csv('./Data/Seoul2/merge_data.csv')
print(data)

view_nan(data)

spring = data[data['season']==0]
print(spring)

summer = data[data['season']==1]
print(summer)

fall = data[data['season'] == 2]
print(fall)

winter = data[data['season'] == 3]
print(winter)

# spring.to_csv('./data/Seoul2/Seoul_spring.csv',index=False)
# summer.to_csv('./data/Seoul2/Seoul_summer.csv',index=False)
# fall.to_csv('./data/Seoul2/Seoul_fall.csv',index=False)
# winter.to_csv('./data/Seoul2/Seoul_winter.csv',index=False)