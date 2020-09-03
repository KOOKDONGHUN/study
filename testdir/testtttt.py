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
tablename = 'KDH_Certificate2-1'

# 내가 원하는 답변자
i_want_writer = '무꿈 님 답변'

detail_address="D:/Study/ANSWERBOT_Project/data/"

que_title_selestor_ls=['div.c-heading._questionContentsArea.c-heading--default-old', # 기본
                         'div.c-heading._questionContentsArea.c-heading--default', # 질문 세부내용이 없는경우
                         'div.c-heading._questionContentsArea.c-heading--multiple']

# 2. 긁어온 url 통해서 자료추출
data = pd.read_csv(f"{detail_address}urls_{name}.txt", sep=',',header=None, encoding='utf-8')

page_nums = data[0].values.tolist()
urls = data[1].values.tolist()

# db없으면 생성
create_table(tablename)

def split_page_num(page_num):
    
    pages = list()
    nums = list()

    for n in page_num:
        temp = n.split('_')
        pages.append(int(temp[0]))
        nums.append(int(temp[1]))

    pages = list(set(pages))
    nums = list(set(nums))

    pages.sort()
    nums.sort()

    return pages, nums

pages, nums = split_page_num(page_nums)


for page in pages:
    for num in nums:
        page_num = '_'.join([str(page),str(num)])
        # print(page_num)