import pandas as pd
import numpy as np

''' pd.DataFrame([Series,Series, ... , 리스트를 포함하는 딕, ...]) 
    Series의 values(list)의 길이나 딕의 리스트의 길이는 동일해야한다.    '''

data = {"fruits" : ["apple","orange", "banana","strawberry","kiwifruit"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(f"{'-'*33}\n※  Origin Data\ndf : \n{df}\n")

df_1 = df.drop(range(0,2)) # 0행 부터 1행을 날린다.
print(f"{'-'*33}\n※  0 ~ 1 raw Drop\ndf_1 : \n{df_1}\n")

df_2 = df.drop('year',axis=1) # year열의 모든행(axis=1) 날린다.
print(f"{'-'*33}\n※  year col Drop\ndf_2 : \n{df_2}\n")


print("-"*33)
np.random.seed(0)
columns = ["apple","orange", "banana","strawberry","kiwifruit"]

df_sec = pd.DataFrame()
for col in columns : # 이렇게 하면 과일의 각 칼럼당 1~10사이의 값 중에 랜덤으로 10행이 생기는듯?
        df_sec[col] = np.random.choice(range(1,11),10) 
# print(df) # 맞고

df_sec.index = range(1,11) # index replace

print(f"{'-'*33}\n※  Origin Data\ndf_sec : \n{df_sec}\n")

print(f"{'-'*33}\n※  Drop even raw\ndf_sec.drop(np.arange(2,11,2)) : \n{df_sec.drop(np.arange(2,11,2))}\n")
# np.arange(2,11,2) -> 2 ~ 10 까지 2의 간격의 리스트

print(f"{'-'*33}\n※  Drop strawberry col\ndf_sec.drop('strawberry',axis=1) : \n{df_sec.drop('strawberry',axis=1)}\n")