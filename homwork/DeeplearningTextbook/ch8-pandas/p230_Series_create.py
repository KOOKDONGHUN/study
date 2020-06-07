import pandas as pd

print('''※ basic create Series\n''')
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12 ,3]

# series = pd.Series(data,index)
fir_series = pd.Series(data,index=index)

print(f"fir_series : \n{fir_series}\n")
