import pymssql as ms

def select_data(tablename, col1, col2, col3, col4):
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    # conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'SELECT {col1}, {col2}, {col3}, {col4} FROM {tablename}'
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

tablename = 'KDH_Certificate'
col_ls = ['id', 'que', 'que_detail', 'ans_detail']

origin_data = select_data(tablename, col1='id', col2='que', col3='que_detail', col4='ans_detail')

def replace_str(data):
    data = data.replace('답변 - ','')
    data = data.replace('질문 - ','')
    data = data.replace(' - ','')
    data = data.replace('..','')
    data = data.replace('...','')
    data = data.replace('....','')
    data = data.replace('??','?')
    data = data.replace('??','?')
    data = data.replace('!','')
    data = data.replace('(내공100)','')


    data = data.replace('\n',' ')
    data = data.replace('//','')
    data = data.replace('ㅠ','')
    data = data.replace('ㅠㅠ','')
    data = data.replace('ㅋ','')

    data = data.replace('  ',' ')
    data = data.replace('  ',' ')

    return data

def convert_dict(origin_data):

    data_dic = dict()

    col1_ls = list()
    col2_ls = list()

    for data in origin_data:
        # print(data[0])
        if data[2] == 'Null':
            t1 = replace_str(data[1])
            if len(t1) > 200 or len(t1) <= 5:
                col1_ls.append('')
            else : 
                col1_ls.append(t1[:])

        elif data[2] != 'Null':
            t2 = data[2]
            t2 = replace_str(t2)
            if len(t2) > 200 or len(t2) <= 5:
                col1_ls.append('')
            else : 
                col1_ls.append(t2[:])
        t3 = data[-1]
        t3 = replace_str(t3)
        if len(t3) > 200 or len(t3) <= 5:
            col2_ls.append('')
        else : 
            col2_ls.append(t3[:])
    
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

df = pd.DataFrame(converted_origin_data)
df.to_csv('./ANSWERBOT_Project/data/ChatbotData.csv',index=None)