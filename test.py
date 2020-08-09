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
        if cnt >= 10 :
            break
        row = cursor.fetchone()
        rows.append(row)
        cnt += 1
    conn.close()

#     print(rows)
#     print(len(rows))

    return rows[:]

tablename = 'KDH_Certificate'
col_ls = ['id', 'que', 'que_detail', 'ans_detail']

# origin_data = select_data(tablename, col1='id', col2='que', col3='que_detail', col4='ans_detail')

def replace_str(data):
    data = data.replace('\n',' ')
    data = data.replace('//','')
    data = data.replace('ㅠ','')
    data = data.replace('ㅋ','')

    return data

# for data in origin_data:
#         # print(f'data : {data}')
#         for col in data:
#                 col = replace_str(col)
#                 temp = col.split('.')
#                 # print(col)
#                 print(temp[:2])


# ls = ['qwe', 'asd']
# a = ' '.join(ls)
# print(a)


import pandas as pd

data = pd.read_csv('./ANSWERBOT_Project/data/ChatbotData2.csv',index_col=0)
print(data)
ls = ['Q', 'A']

max = 0

for col in ls:
        for idx in data[col]:
                idx = idx.split(' ')
                if len(idx) >= max :
                        max = len(idx)

print(max)