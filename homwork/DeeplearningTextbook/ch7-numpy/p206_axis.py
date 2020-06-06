''' 행이 axis = 1
    열이 axis = 0 '''
import numpy as np

arr = np.array([[1,2,3],[4,5,6]])

print(arr.sum())
print(arr.sum(axis=0)) # 열의 총합
print(arr.sum(axis=1)) # 행의 총합

print()

# 행의 합을 구하시오!!
arr2 = np.array([[1,2,3],[4,5,12],[15,20,22]])
print(arr2.sum(axis=1))