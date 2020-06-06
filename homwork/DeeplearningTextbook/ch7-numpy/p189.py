import numpy as np

# 1-dimention calc ex1) use list
storages = [1, 2, 3, 4]
new_storages = []
for n in storages:
    n += n
    new_storages.append(n)

print(f"new_storages : {new_storages}") # new_storages : [2, 4, 6, 8]

# 1-dimention calc ex2) use numpy
storages = np.array(storages)
storages += storages
print(f"storages : {storages}") # storages : [2 4 6 8]