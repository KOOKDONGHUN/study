import numpy as np

''' 모든 요소에 대한 동일한 처리 -> 브로드 캐스트'''
x = np.arange(6).reshape(2,3)
print(x+1)

# 0 ~ 14 사이의 정수값을 갖는 3,5의 ndarray 배열 생성
x = np.arange(15).reshape(3,5)

# 0 ~ 4 사이의 정수값을 갖는 1,5의 ndarray 배열 생성
y = np.array([np.arange(5)])

z = x - y
print(x,"\n")
print(y,"\n")
print(z,"\n")

np.random.seed(100)

# arr = np.arange(31).reshape(5,3)
arr = np.random.randint(0,31,(5,3))
print(arr,"\n")

arr = arr.transpose()
print(arr,"\n")

# arr1 = arr[np.array[1,4,5]] # 행인덱스로 원하는 행만 뽑는게 있었던거 같은데 잘못봤나
arr1 = arr[:,1:4]
print(arr1,"\n")

arr1.sort(0)
print(arr1,"\n")

print(arr1.mean(axis=0))

