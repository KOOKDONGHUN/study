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


######틀린 코드 반성하자 전체 데이터에 대한 중간값에 대해서 이상치를 구하는 것이기 때문에 반복문을 이용하여 칼럼별로 구해주는 것이 맍는 정답이다.
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

# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구하시오.
# 데이터를 어떻게 넣을 것 인가 행데이터 혹은 열데이터로 넣을 것이냐

def outliers(data_out):
    outliers = []
    print(data_out.shape[0])
    for i in range(data_out.shape[0]):
        data = data_out[i, :]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        print(f'lower_bound : {lower_bound}')
        print(f'upper_bound : {upper_bound}\n')

        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return outliers

a = np.array([[1, 100, 150, 5000],[200, 8, 150, 500],[2, 4, 3000, 200]])
print(a)


b = outliers(a)
print(b)
# [(array([1], dtype=int64),), (array([0], dtype=int64),)]