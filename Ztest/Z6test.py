import pandas as pd
import numpy as np

boston = pd.read_csv('./data/csv/boston_house_prices.csv',sep=',',header=1)

boston = boston.describe()


for col in boston.columns :
    print(col)
    print(boston[col])