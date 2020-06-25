import pandas as pd
import numpy as np
from hamsu import view_nan
import time

train_dst = pd.read_csv('./data/dacon/comp1/train_dst.csv', index_col=0, header=0)
test_dst = pd.read_csv('./data/dacon/comp1/test_dst.csv', index_col=0, header=0)

train_src = pd.read_csv('./data/dacon/comp1/train_src.csv', index_col=0, header=0)
test_src = pd.read_csv('./data/dacon/comp1/test_src.csv', index_col=0, header=0)

train_rho = pd.read_csv('./data/dacon/comp1/train_rho.csv', index_col=0, header=0)
test_rho = pd.read_csv('./data/dacon/comp1/test_rho.csv', index_col=0, header=0)

train_y = pd.read_csv('./data/dacon/comp1/train_y.csv', index_col=0, header=0)


def outliers(data, axis= 0):
    import numpy as np
    import pandas as pd
    if type(data) == pd.DataFrame:
        data = data.values
    if len(data.shape) == 1:
        quartile_1, quartile_3 = np.percentile(data,[25,75])
        # print("1사분위 : ", quartile_1)
        # print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
        upper_bound = quartile_3 + (iqr * 1.5)  ## 위
        return np.where((data > upper_bound) | (data < lower_bound))
    else:
        output = []
        for i in range(data.shape[axis]):
            if axis == 0:
                quartile_1, quartile_3 = np.percentile(data[i, :],[25,75])
            else:
                quartile_1, quartile_3 = np.percentile(data[:, i],[25,75])
            # print("1사분위 : ", quartile_1)
            # print("3사분위 : ", quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
            upper_bound = quartile_3 + (iqr * 1.5)  ## 위
            if axis == 0:
                output.append(np.where((data[i, :] > upper_bound) | (data[i, :] < lower_bound))[0])
            else:
                output.append(np.where((data[:, i] > upper_bound) | (data[:, i] < lower_bound))[0])
    return np.array(output)

###
for i in range(len(train_dst.columns)):
    # print('i',i)
    # print('ddd',train_dst.shape[1])
    x=outliers(train_dst.iloc[:,i])#열 별
    rows = list(x[0])#열 별로 값 가져옴
    for idx_col ,j in enumerate(rows):
        train_dst.iloc[j,i]=np.nan

###
train_dst = train_dst.dropna()

###
idx = train_dst.index

train_fu_r = []
train_fu_j = []

for i in range(train_dst.values.shape[0]):
    fu_r = np.fft.fft(train_dst.iloc[i] - train_dst.iloc[i].mean(), n=60).real
    fu_j = np.fft.fft(train_dst.iloc[i] - train_dst.iloc[i].mean(), n=60).imag

    train_fu_r.append(fu_r)
    train_fu_j.append(fu_j)

train_fu_r = pd.DataFrame(train_fu_r,columns=list(np.arange(60)),index=idx)
train_fu_j = pd.DataFrame(train_fu_j,columns=list(np.arange(60)),index=idx)


###
idx = test_dst.index

test_fu_r = []
test_fu_j = []

for i in range(test_dst.values.shape[0]):
    fu_r = np.fft.fft(test_dst.iloc[i] - test_dst.iloc[i].mean(), n=60).real
    fu_j = np.fft.fft(test_dst.iloc[i] - test_dst.iloc[i].mean(), n=60).imag

    test_fu_r.append(fu_r)
    test_fu_j.append(fu_j)

test_fu_r = np.array(test_fu_r)
test_fu_j = np.array(test_fu_j)


test_fu_r = pd.DataFrame(test_fu_r,columns=list(np.arange(60)),index=idx)
test_fu_j = pd.DataFrame(test_fu_j,columns=list(np.arange(60)),index=idx)

###
train = pd.merge(train_rho,train_src,on='id',how='outer')
train = train.dropna()
train = pd.merge(train,train_dst,on='id',how='outer')
train = train.dropna()
train = pd.merge(train,train_fu_r,on='id')
train = pd.merge(train,train_fu_j,on='id')
train = pd.merge(train,train_y,on='id',how='outer')
train = train.dropna()

test = pd.merge(test_rho,test_src,on='id')
test = pd.merge(test,test_dst,on='id')
test = pd.merge(test,test_fu_r,on='id')
test = pd.merge(test,test_fu_j,on='id')

print(train)
print(test)


###
train.to_csv('./data/dacon/comp1/train_new.csv')
test.to_csv('./data/dacon/comp1/test_new.csv')