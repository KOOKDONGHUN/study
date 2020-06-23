import numpy as np

def outlier(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print(f" 1 / 4 : {quartile_1}")
    print(f" 3 / 4 : {quartile_3}")

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    return np.where((data_out > upper_bound) | (data_out < lower_bound))

a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])
b = outlier(a)

print(f"over value locate : {b}")