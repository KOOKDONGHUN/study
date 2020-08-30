# -*- coding: utf-8 -*-
from urllib.request import urlopen
from urllib.parse import quote
import bs4
import pandas as pd

name = "Certificate"
num_per_page = list(range(1,21))
pages = list(range(1,501))

detail_address="D:/Study/ANSWERBOT_Project/data/"

#1. url 긁어오기
for page in pages:
    url = f"https://kin.naver.com/userinfo/answerList.nhn?u=%2BiGFfCq9V7ttCQMaerGI3Uki53svVwaY1w%2BB6%2BDsu6A%3D&isSearch=true&query={quote('자격증')}&sd=answer&y=0&section=qna&isWorry=false&x=0&page={page}"

    source = urlopen(url).read()
    source_bs4 = bs4.BeautifulSoup(source,"html.parser")

    for num in num_per_page:
                                    # #au_board_list > tr:nth-child(2) > td.title > a
                                    # #contentsOfMykin
        url_under=source_bs4.select(f'#au_board_list > tr:nth-of-type({num}) > td.title > a')[0]["href"]

        with open(f"{detail_address}urls_{name}.txt",'a',encoding='utf-8') as file:
                file.write(f"{page}_{num},")
                file.write(url_under+"\n")

data = pd.read_csv(f"{detail_address}urls_{name}.txt",sep=',')
print(len(data))

data = data.drop_duplicates()
print(len(data))