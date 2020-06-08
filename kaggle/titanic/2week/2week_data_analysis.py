import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


train = pd.read_csv('./data/csv/train.csv')
test = pd.read_csv('./data/csv/test.csv')

''' age에서 nan 값을 채워줘야 하는데 nan 값을 가진 사람들의 특징을 고려해보아야한다.?
    왜 nan 인지?에 대해서 말하는 건가?'''
age_nan_rows = train[train['Age'].isnull()]
print(age_nan_rows.head())