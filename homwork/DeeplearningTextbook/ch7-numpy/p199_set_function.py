import numpy as np

arr1 = [2, 5, 7, 9, 5, 2] # list
arr2 = [2, 5, 8, 3, 1] # list

new_arr1 = np.unique(arr1) # ndarray
print(f"arr1 : {arr1}\nnew_arr1 : {new_arr1}\n")

# 합집합
union_arr = np.union1d(arr2,new_arr1)
print(f"arr2 : {arr2}\nnew_arr1 : {new_arr1}\nunion_arr : {union_arr}\n")

# 교집합
intersect_arr = np.intersect1d(arr2,new_arr1)
print(f"arr2 : {arr2}\nnew_arr1 : {new_arr1}\nintersect_arr : {intersect_arr}\n")

# 차집합
setdiff_arr = np.setdiff1d(arr2,new_arr1)
print(f"arr2 : {arr2}\nnew_arr1 : {new_arr1}\nsetdiff_arr : {setdiff_arr}\n")