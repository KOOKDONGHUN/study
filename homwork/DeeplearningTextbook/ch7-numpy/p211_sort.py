import numpy as np

arr = np.array([15, 30, 5])
arr2 = arr.argsort() # 정렬된 배열의 인덱스를 반환한다.
print(arr2,"\n")

arr = np.array([[8,4,2],[3,5,1]])
# print(f"arr : \n{arr}\nnew_arr_argsort : \n{arr.argsort()}")
# print("-"*33)
# print(f"arr : \n{arr}\nnew_arr_npsort : \n{np.sort(arr)}")
# print("-"*33)

# 나머지는 참조하는 배열에 영향을 미치지 않지만 이녀석은 참조하는 배열에 영향을 미친다?
arr2 = arr.sort(1)
print(f"arr : \n{arr}\nnew_arr_sort : \n{arr2}") 