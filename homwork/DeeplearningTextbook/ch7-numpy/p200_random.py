import numpy as np
# from numpy import random
from numpy.random import randint

# arr1 = np.random.randint(0,10,(5,2))
arr1 = randint(0,10,(5,2))
print(f"arr1 : {arr1}")

# arr2 = np.random.randint(0,1,3)
arr2 = np.random.rand(3)
print(f"arr2 : {arr2}")