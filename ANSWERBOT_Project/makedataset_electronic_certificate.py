import pandas as pd

# data = pd.read_csv("./ANSWERBOT_Project/data/ChatbotData_computer.csv",sep=',')
data = pd.read_csv("./ANSWERBOT_Project/data/ChatbotData_electrnic.csv",sep=',')
print(len(data['A']))

data_a = data['A'].drop_duplicates()
print(len(data_a))


print(len(data['Q']))

data_q = data['Q'].drop_duplicates()
print(len(data_q))