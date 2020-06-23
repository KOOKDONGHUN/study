import numpy as np

""" def outlier(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print(f" 1 / 4 : {quartile_1}")
    print(f" 3 / 4 : {quartile_3}")

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    print(data_out > upper_bound)
    print(data_out < lower_bound)

    return np.where((data_out > upper_bound) | (data_out < lower_bound))

a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])
b = outlier(a)

print(f"over value locate : {b}") # over value locate : (array([4, 7], dtype=int64),)
''' index 4, 7 -> over (10000, 5000) '''

# 2개의 컬럼이 10개의 데이터를 가지고있다.
a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
              [1, 20, 3, 4, 1000, 6, 70, 500, 90, 100]])

a2 = np.array([[1, 5000],[200, 8],[2, 4],[3, 7],[8, 2]])

print(a.shape)
print(a2.shape)

b = outlier(a)
print(f"over value locate : {b}")

b = outlier(a2)
print(f"over value locate : {b}") """



def outlier(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print(f" 1 / 4 : {quartile_1}")
    print(f" 3 / 4 : {quartile_3}")

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    print(data_out > upper_bound)
    print(data_out < lower_bound)

    return np.where((data_out > upper_bound) | (data_out < lower_bound))

a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])
b = outlier(a)

print(f"over value locate : {b}") # over value locate : (array([4, 7], dtype=int64),)
''' index 4, 7 -> over (10000, 5000) '''

# 2개의 컬럼이 10개의 데이터를 가지고있다.
a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
              [1, 20, 3, 4, 1000, 6, 70, 500, 90, 100]])

a2 = np.array([[1, 5000],[200, 8],[2, 4],[3, 7],[8, 2]])

print(a.shape)
print(a2.shape)

b = outlier(a)
print(f"over value locate : {b}")

b = outlier(a2)
print(f"over value locate : {b}")