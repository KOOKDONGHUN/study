
import pandas as pd

name = "Certificate"

detail_address="D:/Study/ANSWERBOT_Project/data/"


data = pd.read_csv(f"{detail_address}urls_{name}.txt",sep=',')
print(len(data))

data = data.drop_duplicates()
print(len(data))