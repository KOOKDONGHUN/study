import pandas as pd
from pandas import Series, DataFrame

attri_data1 = { 'ID' : ["100", "101", "102", "103", "104", "106", "108" ],
                'city' : ['서울', '부산', '대전', '광주', '서울', '서울', '서울'],
                'birth_year' : [1990, 1989, 1992, 1997, 1982, 1991, 1988],
                'name' : ['명이', '순돌', '짱구', '태양', '션', '유리', '현아']}
attri_data_frame1 = DataFrame(attri_data1)

print(f'{"-"*33}\n dataframe1 : \n{attri_data_frame1}\n')

attri_data2 = { 'ID' : ['107', '109'],
                'city' : ['봉화', '전주'],
                'birth_year' : [1994, 1988]}
attri_data_frame2 = DataFrame(attri_data2)

print(f'{"-"*33}\n dataframe2 : \n{attri_data_frame2}\n')

# attri_data_frame1.append(attri_data_frame2).sort_values(by='ID',ascending=True)#.reset_index(drop=True)
# print(f'{"-"*33}\n dataframe1 : \n{attri_data_frame1}\n')

# attri_data_frame1 = attri_data_frame1.append(attri_data_frame2).sort_values(by='ID',ascending=True)#.reset_index(drop=True)
# print(f'{"-"*33}\n dataframe1 : \n{attri_data_frame1}\n')

attri_data_frame1 = attri_data_frame1.append(attri_data_frame2).sort_values(by='ID',ascending=True).reset_index(drop=True)
print(f'{"-"*33}\n dataframe1 : \n{attri_data_frame1}\n')