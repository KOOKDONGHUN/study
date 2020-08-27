import pandas as pd

data = pd.read_csv("D:/Study/ANSWERBOT_Project/data/urls_Certificate.txt",sep=',')
print(data)

data = data.drop_duplicates()
print(data)