import pandas as pd
import numpy as np

train = pd.read_csv('c:/titanic/train.csv)
test = pd.read_csv('c:/titanic/test.csv')

train.head()

print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())