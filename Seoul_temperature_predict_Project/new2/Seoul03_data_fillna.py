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

data = data.interpolate()
view_nan(data)
data = data.fillna(method='bfill')
data = data.round(1)
# data.to_csv('./Data/Seoul2/merge_data.csv',index=False)