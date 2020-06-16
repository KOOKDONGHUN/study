import pandas as pd

rain = pd.read_csv('./data/Seoul/Seoul_temp_2010-2020_season.csv',encoding='CP949',header=6,sep=',',error_bad_lines=False)
print(rain)