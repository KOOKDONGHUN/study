import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print()
print(dataset)

''' (seq - size + 1, size) shape의 데이터 셋이 나옴  '''