import numpy as np
import pandas as pd


samsung = pd.read_csv('./data/csv/samsung.csv',
                            index_col = 0,
                            header=0,
                            sep=',',
                            encoding='CP949')

hite = pd.read_csv('./data/csv/jinlo.csv',
                          index_col = 0,
                          header=0,
                          sep=',',
                          encoding='CP949')

# print(samsung)
# print(hite.head())
# print(samsung)
# print(samsung.shape)

# nan 제거
samsung = samsung.dropna(axis=0)
# print(samsung)
# print(samsung.shape)

hite = hite.fillna(method='bfill')
hite = hite.dropna(axis=0)

# nan제거 2
# hite = hite[0:509]

# 판다스에서는 아레의 방식으로 주로 사용함
# hite.loc[0, 1:5] = ['10','20', '30', '40']
# hite.loc['2020-06-02', '고가': '거래량'] = ['10','20','30','40']
# print(hite)

# 삼성과 하이트의 정렬을 오름차순으로 (최근데이터가 아레쪽으로 가도록)
samsung = samsung.sort_values(['일자'], ascending=['True']) #
hite = hite.sort_values(['일자'], ascending=['True']) #

# print(samsung)
print(hite)

# 콤마 제거, 문자를 정수로 변환
for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))

print(samsung)
print(type(samsung.iloc[0,0])) # 

print(hite.isnull().values.any())
print(hite.isnull().any())

print(hite)

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',""))

print(hite)
print(type(hite.iloc[1,1]))

print(samsung.shape)
print(hite.shape)

samsung = samsung.values
hite = hite.values

np.save('./data/samsung2.npy', arr=samsung)
np.save('./data/hite2.npy', arr=hite)

''' scikit-learn    0.22.1 '''