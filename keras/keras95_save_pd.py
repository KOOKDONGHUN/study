import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris.csv',
                        index_col = None,
                        header=0,
                        sep=',')

print(type(datasets))
print(datasets.head())
print(datasets.tail())

print(datasets.values) # 판다스를 넘파이로 바꾸는 키워드

data = datasets.values
print(type(data))

np.save('./data/iris_data.npy',arr = data)