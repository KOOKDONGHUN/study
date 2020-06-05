''' 데이터의 y 값의 종류의 학습 데이터가 고르게 있지 않는 문제 '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv("./data/csv/winequality-white.csv",sep=';',header=0)

data_count = wine.groupby('quality')['quality'].count() # 종류별로 모아준다고?

print(data_count)
'''
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
data_count.plot()
plt.show()

''' 분류를 축소 시킨다? '''