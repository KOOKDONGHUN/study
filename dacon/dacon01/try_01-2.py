import pandas as pd
import numpy as np
from hamsu import view_nan

###
def drop_650_700(_dst_,train,test):
    drop_label = []
    for i in range(5):
        drop_label.append('6' + f'{i+5}' + '0_' + _dst_)
    drop_label.append('700_' + _dst_)

    for i in drop_label:
        train = train.drop(i,axis=1)
        test = test.drop(i,axis=1)

    return train, test

###
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col=0, header=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col=0, header=0)

###
train_dst = train.filter(regex='_dst$', axis=1) #.replace(0, np.NaN)
test_dst = test.filter(regex='_dst$', axis=1) #.replace(0, np.NaN)

train_dst = train_dst.interpolate(axis=1)
test_dst = test_dst.interpolate(axis=1)

train_dst, test_dst = drop_650_700('dst',train_dst,test_dst)

###
train_src = train.filter(regex='_src$', axis=1)
test_src = test.filter(regex='_src$', axis=1)

train_src, test_src = drop_650_700('src',train_src,test_src)

###
train_rho = train.filter(regex='rho$', axis=1)
test_rho = test.filter(regex='rho$', axis=1)

###
train_1 = train.filter(regex='hhb$', axis=1)
train_2 = train.filter(regex='hbo2$', axis=1)
train_3 = train.filter(regex='ca$', axis=1)
train_4 = train.filter(regex='na$', axis=1)

train = pd.merge(train_1,train_2,on='id')
train = pd.merge(train,train_3,on='id')
train = pd.merge(train,train_4,on='id')

###
train_dst.to_csv('./data/dacon/comp1/train_dst.csv')
test_dst.to_csv('./data/dacon/comp1/test_dst.csv')

train_src.to_csv('./data/dacon/comp1/train_src.csv')
test_src.to_csv('./data/dacon/comp1/test_src.csv')

train_rho.to_csv('./data/dacon/comp1/train_rho.csv')
test_rho.to_csv('./data/dacon/comp1/test_rho.csv')

train.to_csv('./data/dacon/comp1/train_y.csv')