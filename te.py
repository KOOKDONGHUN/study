import pandas as pd

df_1 = pd.read_csv('./ANSWERBOT_Project/data/ChatbotData1.csv')
df_2 = pd.read_csv('./ANSWERBOT_Project/data/ChatbotData2.csv')

df_12_axis0 = pd.concat([df_1, df_2])

df_12_axis0.to_csv('./ANSWERBOT_Project/data/ChatbotData3.csv')