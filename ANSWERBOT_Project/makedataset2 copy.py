import pandas as pd

data = pd.read_csv('./ANSWERBOT_Project/data/ChatbotData.csv')
print(data)

data = data.dropna()
# print(data)

data = data.drop_duplicates()
print(data)

# A_data = data.drop_duplicates()
# print(A_data)

data.to_csv('./ANSWERBOT_Project/data/ChatbotData2.csv')