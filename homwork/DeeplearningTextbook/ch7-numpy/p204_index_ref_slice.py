import numpy as np

arr = np.array([[1,2,3],[4,5,6]])
print(f"arr : \n{arr}\narr[1] : \n{arr[1]}")
print("-"*33)
print(f"arr : \n{arr}\narr[1,2] : \n{arr[1,2]}")
print("-"*33)
print(f"arr : \n{arr}\narr[1,1:]: \n{arr[1,1:]}")

print("-"*33)

arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"select 3 : \n{arr2[0,2]}")
print("-"*33)
print(f'''select [[4 5]
        [7 8]]
        
{arr2[1:,:-1]}''')