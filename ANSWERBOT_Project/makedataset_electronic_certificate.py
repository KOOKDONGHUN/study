import pandas as pd

# data = pd.read_csv("./ANSWERBOT_Project/data/ChatbotData_computer.csv",sep=',')
data = pd.read_csv("./ANSWERBOT_Project/data/ChatbotData_electrnic.csv",sep=',')
print(data['A'])

data = data['A'].drop_duplicates()
print(data)

print(__file__)