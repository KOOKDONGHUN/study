import pandas as pd
import numpy as np


# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv')
test = pd.read_csv('./data/dacon/comp1/test.csv')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv')

train = train.loc[:, 'rho':'990_dst']
test = test.loc[:, 'rho':'990_dst']

train_dst = []
train_src = []

test_dst = []
test_src = []

max_test = 0
max_train = 0
train_fu_real = []
train_fu_imag = []
test_fu_real = []
test_fu_imag = []

for i in range(10000):
    tmp_x = 0
    tmp_y = 0
    for j in range(35):
        # if train_src[i, j] == 0:
        #     tmp_x += 1
        #     train_src[i,j] = 0
        #     train_dst[i,j] = 0
        if train_src[i, j] - train_dst[i, j] < 0:
            train_src[i,j] = train_dst[i,j] 
        # if test_src[i, j] == 0:
        #     tmp_y += 1
        #     test_src[i,j] = 0
        #     test_dst[i,j] = 0
        if test_src[i, j] - test_dst[i, j] < 0:
            test_src[i,j] = test_dst[i,j]
    if tmp_x > max_train:
        max_train = tmp_x
    if tmp_y > max_test:
        max_test = tmp_y
    train_fu_real.append(np.fft.fft(train_dst[i]-train_dst[i].mean()).real)
    train_fu_imag.append(np.fft.fft(train_dst[i]-train_dst[i].mean()).imag)
    test_fu_real.append(np.fft.fft(test_dst[i]-test_dst[i].mean()).real)
    test_fu_imag.append(np.fft.fft(test_dst[i]-test_dst[i].mean()).imag)

small = 1e-30

x_train = np.concatenate([train.values[:,0:1]**2,train_dst, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag] , axis = 1)
x_pred = np.concatenate([test.values[:,0:1]**2,test_dst, test_src-test_dst, test_src/(test_dst+small), test_fu_real, test_fu_imag], axis = 1)