import pandas as pd
import numpy as np
from hamsu import view_nan
import time
from sklearn.preprocessing import StandardScaler

###
scaler = StandardScaler()

###
train_dst = pd.read_csv('./data/dacon/comp1/train_dst.csv', index_col=0, header=0)
test_dst = pd.read_csv('./data/dacon/comp1/test_dst.csv', index_col=0, header=0)

train_src = pd.read_csv('./data/dacon/comp1/train_src.csv', index_col=0, header=0)
test_src = pd.read_csv('./data/dacon/comp1/test_src.csv', index_col=0, header=0)

train_rho = pd.read_csv('./data/dacon/comp1/train_rho.csv', index_col=0, header=0)
test_rho = pd.read_csv('./data/dacon/comp1/test_rho.csv', index_col=0, header=0)

train_y = pd.read_csv('./data/dacon/comp1/train_y.csv', index_col=0, header=0)

train_dst_col = train_dst.columns
test_dst_col = test_dst.columns

###
train_damp = np.exp(np.pi*((25 - train_rho.values)/3.44))

train_dst = train_dst.values / train_damp

# train_dst = pd.DataFrame(train_dst, columns=train_dst_col)

###
test_damp = np.exp(np.pi*((25 - test_rho.values)/3.44))

test_dst = test_dst.values / test_damp

# test_dst = pd.DataFrame(test_dst, columns=test_dst_col)

###
train_src = train_src.values
test_src = test_src.values

train_fu_real = []
train_fu_imag = []
test_fu_real = []
test_fu_imag = []

fu_time = 70

for i in range(10000):
    for j in range(29):
        if train_src[i, j] - train_dst[i, j] < 0:
            train_src[i,j] = train_dst[i,j]

        if test_src[i, j] - test_dst[i, j] < 0:
            test_src[i,j] = test_dst[i,j]

    train_fu_real.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1])[0], n=fu_time).real)
    train_fu_imag.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1])[0], n=fu_time).imag)
    test_fu_real.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1])[0], n=fu_time).real)
    test_fu_imag.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1])[0], n=fu_time).imag)


###
train_rho = train_rho.values
test_rho = test_rho.values

train_y = train_y.values

small = 1e-20 # 0인 놈들을 이거로 대체해줌

train = np.concatenate([train_rho[:,0:1]**2, train_dst, train_dst*train_damp, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag, train_y] , axis = 1)
pred = np.concatenate([test_rho[:,0:1]**2,train_dst, test_dst*test_damp, test_src-test_dst, test_src/(test_dst+small),test_fu_real,test_fu_imag], axis = 1)

print(train.shape)
print(pred.shape)

np.save('./data/dacon/comp1/train.npy', arr=train)
np.save('./data/dacon/comp1/pred.npy', arr=pred)