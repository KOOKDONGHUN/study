import pymssql as ms
import kss

def select_data(tablename, col2, col3, col4):
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    # conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'SELECT {col2}, {col3}, {col4} FROM {tablename}'
    cursor.execute(sql)
    rows = list()
    row = cursor.fetchone()
    rows.append(list(row))
    cnt = 1
    while row :
        # if cnt > 10 :
        #     break
        row = cursor.fetchone()
        if row == None :
            break
        else :
            rows.append(list(row))
            cnt += 1
    conn.close()

    # print(rows[-1])
    print(len(rows))
    # print(cnt)

    if rows[-1] == None :
        return rows[:-1]

    rows = sum(rows,[])
    return rows

tablename = 'KDH_Certificate'
col_ls = ['id', 'que', 'que_detail', 'ans_detail']

# origin_data1 = select_data('KDH_Certificate', col2='que', col3='que_detail', col4='ans_detail')
origin_data2 = select_data('KDH_Certificate2', col2='que', col3='que_detail', col4='ans_detail')
origin_data3 = select_data('KDH_Certificate3', col2='que', col3='que_detail', col4='ans_detail')
def to_list(tp):
    temp = list()
    print(tp)
    for t in tp:
        a = list(t)
        temp.append(a)
    return temp

# origin_data1 = to_list(origin_data1)
# print(origin_data1)
# print(origin_data1[0])
# print(type(origin_data1))
# print(type(origin_data1[0]))

# origin_data2 = to_list(origin_data2)
# origin_data3 = to_list(origin_data3)

def check_duplicate(data):
    temp = list()
    for raw in data:
        r = kss.split_sentences(raw)
        temp.append(r)

    print(temp)
    print(type(temp))
    print(len(temp))
    print(type(temp[0]))
    print(len(temp[0]))
    
    return set(sum(temp, []))

# origin_data1 = check_duplicate(origin_data1)
# for i in origin_data1:
#     with open('./raw.txt','a') as f:
#         f.write(str(i)+'\n')

origin_data2 = check_duplicate(origin_data2)
for i in origin_data2:
    with open('./raw.txt','a') as f:
        f.write(str(i)+'\n')

origin_data3 = check_duplicate(origin_data3)
for i in origin_data3:
    with open('./raw.txt','a') as f:
        f.write(str(i)+'\n')

import pandas as pd

# data = pd.read_csv('./raws.txt', sep='\n', encoding='utf-8')
# print(data)

# df = pd.DataFrame(converted_origin_data)
# df.to_csv('./ANSWERBOT_Project/data/last_data.csv',index=None)