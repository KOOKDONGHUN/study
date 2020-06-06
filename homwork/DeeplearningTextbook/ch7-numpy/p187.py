import numpy as np

''' values = np.array(list) <- basic '''
array1 = np.arange(4)
print(f"1dim ndarray : {array1}")

array_1_dim = np.array([1,2,3,4,5,6,7,8]) # (8, )
print(f"array_1_dim : {array_1_dim.shape}")

array_2_dim = np.array([[1,2,3,4],[5,6,7,8]]) # (2, 4)
print(f"array_2_dim : {array_2_dim.shape}")

array_3_dim = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]) # (2, 2, 2)
print(f"array_3_dim : {array_3_dim.shape}")

