'''adarray 배열을 다른 변수에 그래도 대입한 경우 해당 변수의 값을 변경하면 원해 ndarray배열의 값도 변경됩니다 (파이썬의 리스트와 동일)
   때문에 복사하여 사용해야 한다. 복사 방법은 "values = ndarray.copy()"'''

import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])

print("-"*33)

arr2 = arr1
print(f"\narr1 : {arr1}\narr2 : {arr2}")
arr2[0]=100
print("\narr2[0] inserted 100 !!!")
print(f"\narr1 : {arr1}\narr2 : {arr2}")

print("-"*33)

arr2 = arr1.copy()
print(f"\narr1 : {arr1}\narr2 : {arr2}")
arr2[0]=55
print("\narr2[0] inserted 55 !!!")
print(f"\narr1 : {arr1}\narr2 : {arr2}")

print("-"*33)