import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('./Data/Seoul2/merge_data.csv',index_col=0)
print(data)


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

# data.to_csv('./Data/Seoul2/merge_data.csv')