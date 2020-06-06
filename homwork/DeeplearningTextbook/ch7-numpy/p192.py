import numpy as np

arr = np.arange(10)
print(f"arr : {arr}") # arr : [0 1 2 3 4 5 6 7 8 9]

arr = np.arange(10)
arr[0:3] = 1
print(f"arr : {arr}") # arr : [1 1 1 3 4 5 6 7 8 9]

# Quiz 1
print(f"elements(3,4,5) of arr : {arr[3:6]}") # elements(3,4,5) of arr : [3 4 5]

# Quiz 2
arr[3:6] = 24
print(f"arr : {arr}") # arr : [ 1  1  1 24 24 24  6  7  8  9]
