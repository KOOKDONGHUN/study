import pandas as pd
import numpy as np

data = {"fruits" : ["apple","orange", "banana","strawberry","kiwifruit"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df_fir = pd.DataFrame(data)
print(f"df_fir : \n{df_fir}\n")

print("-"*33)
print(f"df_fir.loc[[1,4],['time','year']] : \n{df_fir.loc[[1,4],['time','year']]}\n")
print(f"df_fir.loc[[3,2,0],['time','year']] : \n{df_fir.loc[[3,2,0],['time','year']]}\n")


print("-"*33)
# print(f"df.loc[[2],['banana','kiwifruit']] : \n{df.loc[[2],[df['banana','kiwifruit']]}\n")
np.random.seed(0)
columns = ["apple","orange", "banana","strawberry","kiwifruit"]

df_sec = pd.DataFrame()
for col in columns : # 이렇게 하면 과일의 각 칼럼당 1~10사이의 값 중에 랜덤으로 10행이 생기는듯?
        df_sec[col] = np.random.choice(range(1,11),10) 
# print(df) # 맞고

df_sec.index = range(1,11) # index replace

print(f"df_sec : \n{df_sec}\n")
print(f"df_sec.iloc[range(1,5),[2,4]] : \n{df_sec.iloc[range(1,5),[2,4]]}") # 큰 차이점은 행번호를 입력할때 인덱스 시작 기준이냐 인덱스 이름기준이냐의 차이
# print(f"df_sec.loc[[2:6],['banana','kiwifruit']] : \n{df_sec.loc[[2:6],['banana','kiwifruit']]}") # 슬라이싱 이거아니고 아레꺼
# print(f"df_sec.loc[[2:6, 'banana','kiwifruit']] : \n{df_sec.loc[[2:6, 'banana','kiwifruit']]}") # 판다스와 넘파이는 슬라이싱 하는게 리스트랑 다르다
print(f"df_sec.loc[range(2,6),['banana','kiwifruit']] : \n{df_sec.loc[range(2,6),['banana','kiwifruit']]}")


print("-"*33)
df_thd = df_fir.iloc[[1, 3], [0, 2]]
print(f"df_thd : \n{df_thd}\n")