import numpy as np
import pandas as pd

index = ["growth", "mission", "ishikawa", "pro"]
data = [5, 7, 26, 1]

# Series를 만들어라.
series = pd.Series(index=index,data=data)
print(f"series :\n{series}\n")

# 인덱스를 알파벳순으로 정렬한 series를 aidemy에 대입하라.
# aidemy = series.sort_values()
aidemy = series.sort_index()
print(f"aidemy :\n{aidemy}")

# 인덱스가 "tutor"고, 데이터가 30인 요소를 series에 추가하시오.
# aidemy1 = {"tutor" : 30} -> 아님
aidemy1 = pd.Series(data=30,index=['tutor'])
aidemy2 = series.append(aidemy1)

print(aidemy1,"\n")
print(aidemy2)

# DataFrame을 생성하고 열을 추가합니다.
df = pd.DataFrame()
for index in index :
    df[index] = np.random.choice(range(1,11),10)

# range(시작행, 종료행-1)
df.index = range(1, 11)

#이 책 오타있네
# loc[]을 사용하여 df의 2~5행(4개 행)과 "banana", "kiwifruit"의 2열을 포함한 DataFrame을 df에 대입하세요.
# 첫 번째 행의 인덱스는 1이며, 이후의 인덱스는 정수의 오름차순입니다.
# aidemy3 = df.loc[2:6, ["banana","kiwifruit"]]
aidemy3 = df.loc[range(2,6), ["ishikawa"]]
print("\n",aidemy3)