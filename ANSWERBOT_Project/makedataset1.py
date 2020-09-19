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
    data = data.replace('답변 -> ','')
    data = data.replace('※ E-TEST','')
    data = data.replace('※ 한국사 문제집','')
    data = data.replace('※ 모스마스터','')
    data = data.replace('www.ybmit.com','')
    data = data.replace('-----------------------------------------------------------------------','')
    data = data.replace('license.kpc.or.kr','')
    data = data.replace('답변 부탁드립니다','')
    data = data.replace('※ kpc 자격','')
    data = data.replace('메인화면','')
    data = data.replace('book.naver.com','')
    data = data.replace('※ ITQ 엑셀 ','')
    data = data.replace('※ 시나공 책','')
    data = data.replace('확실하게 아시는 분들만 답변 해주세요','')
    data = data.replace('(컴맹이고 액셀은 사용할 줄 모름)','')
    data = data.replace('(갱신 필요 없음)','')
    data = data.replace('※ 컴퓨터 분야별 자격증들','')
    data = data.replace('♣IT관련 자격증♣','')
    data = data.replace('www.q-net.or.kr','')
    data = data.replace('※ 시험버전','')
    data = data.replace('(급수 무관)','')
    data = data.replace('만점이 500 이면요?','')
    data = data.replace('※ 신청 방법','')
    data = data.replace('※ 비서1급 필기 과목 ','')
    data = data.replace('※ 비서1급 필기 합격기준','')
    data = data.replace('※ 원서접수 경로','')
    data = data.replace('※ 엑셀로 시험을 보는 자격증 ','')
    data = data.replace('문의번호 : 1577-9402','')
    data = data.replace('※ 준비기간 (하루 3시간 ~ 4시간)','')
    data = data.replace('www.ihd.or.kr','')
    data = data.replace('※ kpc 자격 (itq 자격증 취득 조회)','')
    data = data.replace('※ ITQ 파워포인트 이공자 교재','')
    data = data.replace('※ 난이도 ','')
    data = data.replace('※ MOS 2010','')
    data = data.replace('※ itq 시험','')
    data = data.replace('※ OA 자격증 ','')
    data = data.replace('※ diat 시험','')
    data = data.replace('※ ITQ 엑셀 (2010 버전 시나공)','')
    data = data.replace('※ 컴활2급 (시나공 총정리 책, 실기 기본서 책)','')
    data = data.replace('※','')

    data = data.replace('\n',' ')
    data = data.replace('//','')
    data = data.replace('ㅠ','')
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