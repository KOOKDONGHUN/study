import pandas as pd

''' pd.DataFrame([Series,Series, ... , 리스트를 포함하는 딕, ...]) 
    Series의 values(list)의 길이나 딕의 리스트의 길이는 동일해야한다.    '''

data = {"fruits" : ["apple","orange", "banana","strawberry","kiwifruit"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)


print('''※ Basic create DataFrame\n''')

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]

series1 = pd.Series(data1,index)
series2 = pd.Series(data2,index)

fir_df = pd.DataFrame([series1,series2]) # 행번호 자동생성
print(f"fir_df : \n{fir_df}")

''' DataFrame에서는 행의 번호를 인덱스, 열의 이름을 컬럼이라 칭한다. '''