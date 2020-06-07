import pandas as pd
import numpy as np

data = {"fruits" : ["apple","orange", "banana","strawberry","kiwifruit"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(f"{'-'*33}\n※  Origin Data\ndf : \n{df}\n")

print(f"※  Sort ascending=True by 'year'\ndf_sort_year : \n{df.sort_values(by='year',ascending=True)}\n")

# 순서가 빠른 열이 우선적으로 정렬된다.
print(f"※  Sort ascending=True by ['time','year']\ndf_sort_time_year : \n{df.sort_values(by=['time','year'],ascending=True)}\n")
print(f"※  Sort ascending=True by ['year','time']\ndf_sort_time_year : \n{df.sort_values(by=['year','time'],ascending=True)}\n")


np.random.seed(0)
columns = ["apple","orange", "banana","strawberry","kiwifruit"]
df_sec = pd.DataFrame()
for col in columns :
        df_sec[col] = np.random.choice(range(1,11),10) 
df_sec.index = range(1,11)
print(f"{'-'*33}\n※  Origin Data\ndf_sec : \n{df_sec}\n")

print(f"※  Sort ascending=True by columns\ndf_sec_sort_columns : \n{df_sec.sort_values(by=columns,ascending=True)}\n")
# 순서가 빠른 열이 우선적으로 정렬되기 때문에 apple를 기준으로 정렬된 것을 볼 수 있다.


print(f"{'-'*42}")
print(f"{'-'*15}Filtering{'-'*18}")

print(f"{'-'*42}\n※  Origin Data\ndf : \n{df}\n")
print(f"※  Filtering by even\ndf_even : \n{df.index % 2 == 0}\n")
print(f"※  Filtering by even\ndf_even : \n{df[df.index % 2 == 0]}\n")


print(f"{'-'*42}")
print(f"※  Origin Data\ndf_sec : \n{df_sec}\n")

df_sec = df_sec.loc[ df_sec['apple'] >= 5 ]
print(f"※  Filtring by df_sec['apple'] >= 5\ndf_filtering_apple : \n{df_sec}\n")

df_sec2 = df_sec.loc[ df_sec['kiwifruit'] >= 5 ]
print(f"※  Filtring by df_sec['kiwifruit'] >= 5\ndf_filtering_kiwifruit : \n{df_sec2}\n")

# df_sec = df_sec.loc[ df_sec['kiwifruit'] >= 5 ][ df_sec['apple'] >= 5 ] <- 가능 3개도 가능하겠지 그럼 아마도
df_sec = df_sec.loc[ df_sec['kiwifruit'] >= 5 ]
print(f"※  Filtring by df_sec['apple','kiwifruit'] >= 5\ndf_filtering_2columns : \n{df_sec}\n")
