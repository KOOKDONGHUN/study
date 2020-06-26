import pandas as pd
import numpy as np
from hamsu import view_nan
import time
from sklearn.preprocessing import StandardScaler

train_dst = pd.read_csv('./data/dacon/comp1/train_dst.csv', index_col=0, header=0)
test_dst = pd.read_csv('./data/dacon/comp1/test_dst.csv', index_col=0, header=0)

train_src = pd.read_csv('./data/dacon/comp1/train_src.csv', index_col=0, header=0)
test_src = pd.read_csv('./data/dacon/comp1/test_src.csv', index_col=0, header=0)

train_rho = pd.read_csv('./data/dacon/comp1/train_rho.csv', index_col=0, header=0)
test_rho = pd.read_csv('./data/dacon/comp1/test_rho.csv', index_col=0, header=0)

train_y = pd.read_csv('./data/dacon/comp1/train_y.csv', index_col=0, header=0)

