import numpy as np

arr = np.array([4, -9, 16, -4, 20])
arr_abs = np.abs(arr)
print(f"arr : {arr}\narr_abs : {arr_abs}\n")

arr_exp = np.exp(arr_abs)
print(f"arr : {arr}\narr_exp : {arr_exp}\n")

arr_sqrt = np.sqrt(arr_abs)
print(f"arr : {arr}\narr_sqrt : {arr_sqrt}\n")