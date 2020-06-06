import numpy as np

arr = np.arange(9).reshape(3,3)
print(f"np.dot(arr,arr) : {np.dot(arr,arr)}")

vec = arr.reshape(9)
print(f"vec : {np.linalg.norm(vec)}")