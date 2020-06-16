import pandas as pd

fruits = {"banana" : 3,"orange" : 4, "grape" : 1, "peach" : 5}

series_fruits = pd.Series(fruits)

print("-"*33)
print('''※ add Series elements\n''')
print(f"series_fruits : \n{series_fruits}\n")
print("Append 'Jeju tangerine' !!!!!!!!!!!\n")
series_fruits = series_fruits.append(pd.Series([10],index=["Jeju tangerine"]))
print(f"series_fruits : \n{series_fruits}\n")


print("-"*33)
print('''※ add Series elements\n''')
print(f"series_fruits : \n{series_fruits}\n")
print("Append 'pineapple' !!!!!!!!!!!\n")
pineapple = pd.Series([12],index=["pineapple"])
series_fruits = series_fruits.append(pineapple)
series_fruits = series_fruits.append(pd.Series({"apple": 2})) # -> 이렇게 딕의 형태로 넣어 주는것도 가능하다. 
#                                                                       그렇다면 한번에 여러개 추가 가능할것 같다. 
#                                                                       해보지는 않겠다.
print(f"series_fruits : \n{series_fruits}\n")


print("-"*33)
print('''※ Drop Series elements\n''')
print(f"series_fruits : \n{series_fruits}\n")
print("Drop 'Jeju tangerine' !!!!!!!!!!!\n")
series_fruits = series_fruits.drop(["Jeju tangerine"])
print(f"series_fruits : \n{series_fruits}\n")


print("-"*33)
print('''※ Filtering Series elements ex1\n''')
print(f"series_fruits : \n{series_fruits}\n")
# condition = [True, True, False, False]#, False] -> 인덱스의 개수와 맞아야함 당연한거긴한데 한번해봄 
condition = [True, True, False, False, False, False]
print(f"series_fruits : \n{series_fruits[condition]}\n")

'''리스트컴리핸션할때 반복문뒤에 조건문이 온다 하지만 조건문이 if else라면 조건문이 먼저오고 반복문이 와야한다'''
condition = [True if "apple" in ele else False for ele in series_fruits.index]
print(type(condition))
print(condition,"\n")
print(f"series_fruits : \n{series_fruits[condition]}\n")

print('''※ Filtering Series elements ex2\n''')
print(f"series_fruits : \n{series_fruits}\n")
print(f"series_fruits : \n{series_fruits[series_fruits>=4]}\n")

print('''※ Filtering Series elements Quiz\n''')
print(f"series_fruits : \n{series_fruits}\n")
print(f"series_fruits : \n{series_fruits[series_fruits>=4][series_fruits <=10]}\n")


print("-"*33)
print('''※ Sort Series elements ex\n''')
print(f"series_fruits.sort_index() : \n{series_fruits.sort_index()}\n")
print(f"series_fruits.sort_values(ascending=False) : \n{series_fruits.sort_values(ascending=False)}\n")