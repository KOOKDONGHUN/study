# -*- coding: utf-8 -*-
from urllib.request import urlopen
from urllib.parse import quote
import bs4
import pandas as pd
import pymssql as ms
import time

def insert_db(tablename, insert_data : dict()):
    conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'INSERT INTO {tablename} values(%s, %s ,%s ,%s ,%s)'
    cursor.execute(sql, (insert_data['id'], insert_data['que_title'], insert_data['que_detail'], insert_data['ans_writer'], insert_data['ans_detail']))
    conn.commit()
    conn.close()

def create_table(tablename):
    conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tablename}' AND xtype='U')\
         CREATE TABLE {tablename} (id text null, que text null, que_detail text null,\
             ans_writer text null, ans_detail text null)"
    cursor.execute(sql)
    conn.commit()
    conn.close()

def select_data(tablename):
    conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'SELECT * FROM {tablename}'
    cursor.execute(sql)
    row = cursor.fetchone()
    while row :
        print(row)
        row = cursor.fetchone()
    conn.close()

name = "Certificate"

# 테이블 명
tablename = 'KDH_Certificate2'

# 내가 원하는 답변자
i_want_writer = '무꿈 님 답변'

detail_address="D:/Study/ANSWERBOT_Project/data/"

que_title_selestor_ls=['div.c-heading._questionContentsArea.c-heading--default-old', # 기본
                         'div.c-heading._questionContentsArea.c-heading--default', # 질문 세부내용이 없는경우
                         'div.c-heading._questionContentsArea.c-heading--multiple']

# 2. 긁어온 url 통해서 자료추출
data = pd.read_csv(f"{detail_address}urls_{name}.txt", sep=',',header=None, encoding='utf-8')

page_num = data[0].values.tolist()
urls = data[1].values.tolist()

print(len(page_num)) # 11660

# db없으면 생성
create_table(tablename)

a = 261
b = 20
c = 6

starts = ((a-1)*b) + c - 1 

'''나중에 제거 해야할 토큰 "\xa0"'''
for idx, url in enumerate(urls[starts:]):
    time.sleep(2)
    idx += starts
    print(url)
    url = url.replace('§', f'{quote("§")}')
    print(url)
    insert_data = {'id' : f'{page_num[idx]}',
                'que_title' : 'Null',
                'que_detail' : 'Null',
                'ans_writer' : 'Null',
                    'ans_detail' : 'Null',
                }

    source = urlopen(url.strip()).read()
    source_bs4 = bs4.BeautifulSoup(source,"html.parser")

    # 위에 언급한대로 경우에 따라 질문의 큰 제목이 다름 // 반복
    for selector in que_title_selestor_ls :
        question_selector = f'#content > div.question-content > div > {selector} > div.c-heading__title > div > div.title'
        try :
            insert_data['que_title'] = source_bs4.select(question_selector)[0].text.strip()
        except :
            print(f'{page_num[idx]}--{selector[-8:]} question Title Error !!!!')
            pass

    # 질문 세부내용에 대한 text 추출 이건 에러 발생했다는건 사진만 있거나 내용이 없거나 때문에 그냥 Null 처리
    question_detail_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content'
    try :
        insert_data['que_detail'] = source_bs4.select(question_detail_selector)[0].text.strip()
    except :
        print(f'{page_num[idx]} answer Writer Error !!!!')
        pass
    

    # 왜 그런지는 모르겠는데 내가 선택한 답변자의 답변이 없는 경우 무한 반복을 방지하기 위해 answer_num 변수 선언
    answer_num = 1
    while answer_num <= 10:
        
        # 답변자의 이름 선택자
        answer_writer_selector = f'#answer_{answer_num} > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p'

        try :
            insert_data['ans_writer'] = source_bs4.select(answer_writer_selector)[0].text.strip()
        except:
            print(f'{page_num[idx]} answer Writer Error !!!!')
            pass
            
        # 답변 내용
        # answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content' # 기존에 사용하던거 뒤에 '알아두세요 이게 출력됨'
        # answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user > div' # 채택된 답변일 경우
        answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user' # 위에꺼는 공백을 가져오네 ㅋ...

        try :
            insert_data['ans_detail'] = source_bs4.select(answer_detail_selector)[0].text.strip()
        except:
            print(f'{page_num[idx]} answer Detail Error !!!!')
            pass
        
        # 왜인지는 모르겠으나 비공개 답변인 경우로 답변내용이 추출됨을 방지
        if insert_data['ans_writer'] == i_want_writer and insert_data['ans_writer'] != '비공개 답변':
            
            # text 추출이 완료된 시점에서 db에 insert
            insert_db(tablename, insert_data)

            break # while 탈출 -- 답변에 대한 크롤링 stop

        # while cnt
        answer_num += 1

    print(insert_data)