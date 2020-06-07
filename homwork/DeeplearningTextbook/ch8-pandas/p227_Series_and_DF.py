''' Series와 DataFrame의 데이터 확인 '''

import pandas as pd

''' Series에 딕을 넣으면 키값을 기준으로 오름차순으로 변경
    둘의 차이점은 Series는 라벨이 붙은 1차원의 데이터 형식으로 사용하도록하고
    DataFrame은 Series를 묶은 듯한 2차원 데이터 구조 형식이다. 
    DataFrame구조를 Series로 읽을 경우 에러는 발생하지는 않지만 적절하지 않은 방법으로 데이터를 불러왔다고 봐야할 것 같다. '''

fruits = {"orange":2,
          "banana":3}
print(f"fruits : \n{pd.Series(fruits)}\n")
"""
fruits : 
orange    2
banana    3
dtype: int64"""

data = {"fruits" : ["apple","orange", "banana","strawberry","kiwifruit"],
        "yeat" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(f"DataFrame : \n{df}\n")
"""
DataFrame :
       fruits  yeat  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3"""

print(f"Series_data : \n{pd.Series(data)}\n")
"""
Series_data :
fruits    [apple, orange, banana, strawberry, kiwifruit]
yeat                      [2001, 2002, 2001, 2008, 2006]
time                                     [1, 4, 5, 6, 3]
dtype: object"""

# print(f"DataFrame_fruits : \n{pd.DataFrame(fruits)}\n") # ValueError: If using all scalar values, you must pass an index