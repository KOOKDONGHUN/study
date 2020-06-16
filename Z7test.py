import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime,time


price_df=pd.read_csv("project/pro1/test.csv",index_col=0,header=0)

print(type(price_df))

print("price_df")
print(price_df)
print("price_df.describe()")
print(price_df.describe())
print("price_df.info()")
price_df.info()
print("price_df.shape")
print(price_df.shape)

price_df["month"] =[i[-5:-3] for i in list(price_df.index)]
price_df["year"] =[i[:4] for i in list(price_df.index)]

from sklearn.preprocessing import MinMaxScaler,StandardScaler

# scaler = MinMaxScaler()
# scaler2 = MinMaxScaler()

scaler = StandardScaler()
scaler2 = StandardScaler()

# print(price_df)
# mapping

price_df.loc[:,["snp500","nikkei225","shanghai"]]=scaler.fit_transform(price_df.loc[:,["snp500","nikkei225","shanghai"]])

print(pd.pivot_table(price_df,index=["year","month"]))