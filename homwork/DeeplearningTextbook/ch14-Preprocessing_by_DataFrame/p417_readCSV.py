import pandas as pd

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)  

# 각 수치가 무엇을 나타내는지 칼럼 헤더를 추가합니다.
df.columns = ["", "Alcohol", "Maic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", 'Flavanoids', "Nonflavanoid phenols",
              "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(f"df : \n{df}")