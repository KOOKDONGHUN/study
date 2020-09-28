import pymssql as ms

def select_data(tablename, col2, col3, col4):
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    # conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'SELECT {col2}, {col3}, {col4} FROM {tablename}'
    cursor.execute(sql)
    rows = list()
    row = cursor.fetchone()
    rows.append(row)
    cnt = 1
    while row :
        # if cnt > 10 :
        #     break
        row = cursor.fetchone()
        rows.append(row)
        cnt += 1
    conn.close()

    print(rows[-1])
    print(len(rows))
    print(cnt)

    if rows[-1] == None :
        return rows[:-1]

    return rows

origin_data1 = select_data('KDH_Certificate2', col2='que', col3='que_detail', col4='ans_detail')
origin_data2 = select_data('KDH_Certificate3', col2='que', col3='que_detail', col4='ans_detail')

def replace_str(data):

    data = data.replace('1:1 지정 질문입니다!','')
    data = data.replace('1:1 지정 질문입니다! ?','')
    data = data.replace('1:1 질문을 주셔서 감사합니다.','')
    data = data.replace('1:1 지정 질문입니다.?','')
    data = data.replace('\n',' ')
    data = data.replace('//','')
    data = data.replace('/','')
    data = data.replace('ㅠ','')
    data = data.replace('ㅜ','')
    data = data.replace('ㅠㅠ','')
    data = data.replace('ㅜㅜ','')
    data = data.replace('ㅋ','')
    data = data.replace('..','')
    data = data.replace('...','')
    data = data.replace('  ',' ')
    data = data.replace('  ',' ')
    data = data.replace('  ',' ')
    data = data.replace('  ',' ')
    data = data.replace('\t','')
    data = data.replace('??','')
    data = data.replace('? ','')
    data = data.replace(' ?','')
    data = data.replace(' ? ','')
    data = data.replace('      ','')
    
    
    
    return data

def convert_dict(origin_data):

    data_dic = dict()

    col1_ls = list()
    col2_ls = list()

    for data in origin_data:

        if data[1] == 'Null':
            t1 = replace_str(data[0])
            col1_ls.append(t1[:])

        elif data[1] != 'Null':
            t2 = data[1]
            t2 = replace_str(t2)
            col1_ls.append(t2[:])

        t3 = data[-1]
        t3 = replace_str(t3)
        col2_ls.append(t3[:])
    
    data_dic['Q'] = col1_ls
    data_dic['A'] = col2_ls
    data_dic['label'] = [0 for i in range(len(col2_ls))]

    return data_dic

# for data in origin_data:
#     print(data,'\n')

converted_origin_data1 = convert_dict(origin_data1)
converted_origin_data2 = convert_dict(origin_data2)

# print(converted_origin_data['que'],'\n')
# print(converted_origin_data['ans'])

import pandas as pd

df1 = pd.DataFrame(converted_origin_data1)
df1 = df1.drop_duplicates()
df2 = pd.DataFrame(converted_origin_data2)
df2 = df2.drop_duplicates()
df3 = pd.concat([df1, df2], ignore_index=True)
df3.to_csv('./ANSWERBOT_Project/data/simpledatasets.csv',index=None)