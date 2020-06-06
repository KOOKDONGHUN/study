import numpy as np

arr = np.arange(10).reshape(2,5)
# print(f"arr : \n{arr}\nnew_arr : \n{arr.transpose()}")
print(f"arr : \n{arr}\nnew_arr : \n{arr.T}")