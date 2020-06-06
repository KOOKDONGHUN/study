import numpy as np

arr = np.array([[1,2,3,4],[5,6,7,8]])
print(f"arr : {arr}\n")
print(f"({arr.shape[0]}, {arr.shape[1]})\n")

print("-"*33)

print(f"arr : \n{arr}")
print(f"arr.transpose : \n{arr.transpose()}")

print("-"*33)

print(f"arr : \n{arr}")
arr2 = arr.reshape(4,2)#.copy()
print(f"arr : \n{arr}")
print(f"arr2 : \n{arr2}")