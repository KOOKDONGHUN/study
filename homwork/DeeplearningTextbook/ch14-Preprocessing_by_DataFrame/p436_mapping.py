import pandas as np
from pandas import DataFrame

attri_data1 = { 'ID' : ["100", "101", "102", "103", "104", "106", "108", '110', '111', '113'],
                'city' : ['서울', '부산', '대전', '광주', '서울', '서울', '서울', '대전', '광주', '서울'],
                'birth_year' : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
                'name' : ['명이', '순돌', '짱구', '태양', '션', '유리', '현아', '태식', '민수', '호식']}
attri_data_frame1 = DataFrame(attri_data1)

print(f'{"-"*33}\n dataframe1 : \n{attri_data_frame1}\n')

city_map = {'서울' : '서울',
            '광주' : '전라도',
            '부산' : '경상도',
            '대전' : '충청도'}

print(f"city_map : {city_map}")

attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)

print(f'{"-"*33}\n dataframe1 : \n{attri_data_frame1}\n')