import numpy as np

arr_List = [x for x in range(10)]

# use list copy by slicing
arr_List_2 = arr_List[:]
print(f"\narr_List : {arr_List}\narr_List_2 : {arr_List_2}")

arr_List_2[0] = 100
print(f"\narr_List_2[0] inserted 100 !!!")
print(f"\narr_List : {arr_List}\narr_List_2 : {arr_List_2}")

print("-"*33)

# use ndarray copy by slicing
arr_List = np.arange(10)

arr_List_copy = arr_List[:]
print(f"\narr_List : {arr_List}\narr_List_copy : {arr_List_copy}")

arr_List_copy[0] = 100
print(f"\narr_List_copy[0] inserted 100 !!!")
print(f"\narr_List : {arr_List}\narr_List_copy : {arr_List_copy}")

print("-"*33)

# use ndarray.copy()
arr_List = np.arange(10)

# arr_List_copy = arr_List.copy()
arr_List_copy = arr_List[:].copy()
print(f"\narr_List : {arr_List}\narr_List_copy : {arr_List_copy}")

arr_List_copy[0] = 100
print(f"\narr_List_copy[0] inserted 100 !!!")
print(f"\narr_List : {arr_List}\narr_List_copy : {arr_List_copy}")

''' 삽입과 삭제만 참조하는 인덱스에 영향을 미치는듯 하다 
    transpose나 다른 함수를 사용했을때는 참조하는 배열에 영향을 주지 않는듯 보인다 앞으로 더 해봐야 알것같다'''