import numpy as np

arr = np.arange(15).reshape(3,5)
print(arr.mean(0))
print(arr.sum(1))
print(arr.min())
print(arr.max())

# 각 열에서 최대값의 인덱스 번호를 찾는 방법
print(arr.argmax(axis=0))