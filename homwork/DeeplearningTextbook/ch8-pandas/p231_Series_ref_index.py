import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12 ,3]

fir_series = pd.Series(data,index=index)

print("-"*33)
print('''※ Series ref index ex\n''')
fruits = {"banana" : 3,"orange" : 4, "grape" : 1, "peach" : 5}

series_fruits = pd.Series(fruits)
print(f"series_fruits : \n{series_fruits}\nseries_fruits_slice[0:2] : \n{series_fruits[0:2]}\n")

print(f"series_fruits : \n{series_fruits}\nseries_fruits_slice[['orange','peach']] : \n{series_fruits[['orange','peach']]}\n")


print("-"*33)
print('''※ Series ref index Quiz\n''')
items1 = fir_series[1:4]
print(f"fir_series : \n{fir_series}\nfir_series[1:4] : \n{items1}\n")

items2 = fir_series[["apple", "banana", "kiwifruit"]]
print(f"fir_series : \n{fir_series}\nfir_series[['apple', 'banana', 'kiwifruit']] : \n{items2}\n")
