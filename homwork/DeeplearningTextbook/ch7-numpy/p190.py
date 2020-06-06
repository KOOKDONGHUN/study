import numpy as np

arr = np.array([2, 5, 3, 4, 8])

# arr + arr
print(f"arr + arr = {arr+arr}")

# arr -arr
print(f"arr - arr = {arr-arr}")

# arr ** 3
print(f"arr ** 3 = {arr**3}")

# 1 / arr
print(f"1 / arr = {1/arr}")

"""
arr + arr = [ 4 10  6  8 16]
arr - arr = [0 0 0 0 0]
arr ** 3 = [  8 125  27  64 512]
1 / arr = [0.5        0.2        0.33333333 0.25       0.125     ]"""