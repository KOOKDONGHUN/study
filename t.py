import pandas as pd

data = pd.read_csv('./raw3.txt', sep='\n', encoding='CP949')
print(data)

data = data.drop_duplicates()
print(data)
data.to_csv('./raw3.csv',index=None)
print(__file__)