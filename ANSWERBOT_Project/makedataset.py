# import torch
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

import pymssql as ms

def select_data(tablename, col1, col2, col3, col4):
    conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    # conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'SELECT {col1}, {col2}, {col3}, {col4} FROM {tablename}'
    cursor.execute(sql)
    rows = list()
    row = cursor.fetchone()
    rows.append(row)
    cnt = 0
    while row :
        if cnt >= 100 :
            break
        row = cursor.fetchone()
        rows.append(row)
        cnt += 1
    conn.close()

    print(rows)
    print(len(rows))

    return rows[:-1]

tablename = 'KDH_Certificate'
col_ls = ['id', 'que', 'que_detail', 'ans_detail']

origin_data = select_data(tablename, col1='id', col2='que', col3='que_detail', col4='ans_detail')

def replace_str(data):
    data = data.replace('\n',' ')
    data = data.replace('//','')
    data = data.replace('ㅠ','')
    data = data.replace('ㅋ','')

    return data

def convert_dict(origin_data):

    data_dic = dict()

    col1_ls = list()
    col2_ls = list()

    for data in origin_data:
        # print(data[0])
        if data[2] == 'Null':
            t1 = replace_str(data[1])
            col1_ls.append(t1[:30])
        elif data[2] != 'Null':
            t2 = data[1]+' '+data[2]
            t2 = replace_str(t2)
            col1_ls.append(t2[:30])
        t3 = data[-1]
        t3 = replace_str(t3)
        col2_ls.append(t3[:30])
    
    data_dic['Q'] = col1_ls
    data_dic['A'] = col2_ls
    data_dic['label'] = [0 for i in range(len(col2_ls))]

    return data_dic

# for data in origin_data:
#     print(data,'\n')

converted_origin_data = convert_dict(origin_data)

# print(converted_origin_data['que'],'\n')
# print(converted_origin_data['ans'])

import pandas as pd

# df = pd.DataFrame(converted_origin_data)
# df.to_csv('./Chatbot_data/ChatbotData.csv',index=None)