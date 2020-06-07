'''p240_DataFrame_create.py copy'''

import pandas as pd

''' pd.DataFrame([Series,Series, ... , 리스트를 포함하는 딕, ...]) 
    Series의 values(list)의 길이나 딕의 리스트의 길이는 동일해야한다.    '''

data = {"fruits" : ["apple","orange", "banana"],
        "year" : [2001, 2002, 2001],
        "time" : [1, 4, 5]}

df = pd.DataFrame(data)
print(f"df : \n{df}\n")


print("-"*33)
print('''※ Basic create DataFrame\n''')

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]

series1 = pd.Series(data1,index)
series2 = pd.Series(data2,index)

fur_df = pd.DataFrame([series1,series2]) # 행번호 자동생성
print(f"fur_df : \n{fur_df}\n")

''' DataFrame에서는 행의 번호를 인덱스, 열의 이름을 컬럼이라 칭한다. '''

print("-"*33)
print('''※ Replace DataFrame index\n''')
fur_df.index = [1,2]
print(f"fur_df : \n{fur_df}\n")


''' 데이터 프레임 구조에서 새로운 행을 추할때 컬럼명(열)에 맞게 데이터를 삽입해주면 순서는 바뀌어도 상관없고
    컬럼명이 기존 데이터 프레임에 존재하지 않는다면 새로운 컬럼을 자동으로 생성한다.
    새로운 컬럼이 추가되면 기존 행들(index)에 대한 컬럼값이 nan으로 자동으로 채워진다. '''

print("-"*33)
print('''※ Add DataFrame index(raw)\n''')
series_raw = pd.Series(data=['pineapple',2003,'True',3],index=['fruits','year','Summer','time'])
df = df.append(series_raw,ignore_index=True)
print(f"df : \n{df}\n")


print("-"*33)
print('''※ Add DataFrame colum(col)\n''')
df["price"] = [160,150,140,170] # 행의 갯수에 맞게 넣어줘야함 안그러면 Error!!
print(f"df : \n{df}\n")


print("-"*33)
print('''※ Add DataFrame colum(col) Quiz\n''')
new_col = pd.Series(data=[15,3],index=fur_df.index)
fur_df["pineapple"] = new_col
print(f"fur_df : \n{fur_df}\n")