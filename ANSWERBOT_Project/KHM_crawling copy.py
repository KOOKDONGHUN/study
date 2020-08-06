# -*- coding: utf-8 -*-
from urllib.request import urlopen
from urllib.parse import quote
import bs4

# name = __file__.split("\\")[-1][:-3]
name = "Certificate"
# num_per_page = range(1,21)#한페이지 10개
# pages = range(1,1450)# 10개 페이지

num_per_page = range(1,21)#한페이지 10개
pages = [1]# 10개 페이지

detail_address="D:/Study/ANSWERBOT_Project/data/"

# import urllib.parse

# encode = '%2BiGFfCq9V7ttCQMaerGI3Uki53svVwaY1w%2BB6%2BDsu6A%3D'
# decode = urllib.parse.unquote(encode)
# print(decode)

#1. url 긁어오기
for page in pages:
    url = f"'https://kin.naver.com/userinfo/answerList.nhn?u=%2BiGFfCq9V7ttCQMaerGI3Uki53svVwaY1w%2BB6%2BDsu6A%3D&isSearch=true&query={quote('자격증')}&sd=answer&y=0&section=qna&isWorry=false&x=0&page={page}"
            # 'https://kin.naver.com/userinfo/answerList.nhn?u=%2BiGFfCq9V7ttCQMaerGI3Uki53svVwaY1w%2BB6%2BDsu6A%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page=1'
    # url = f"https://kin.naver.com/search/list.nhn?query={quote('요리')}+{quote('레시피')}&page={page}"
          # https://kin.naver.com/search/list.nhn?query=%EC%9A%94%EB%A6%AC+%EB%A0%88%EC%8B%9C%ED%94%BC
          # 
    source = urlopen(url).read()
    source_bs4 = bs4.BeautifulSoup(source,"html.parser")
#    urls=[]
    for num in num_per_page:
        # contentsOfMykin
        #au_board_list > tr:nth-child(1) > td.title > a
        # # //*[@id="contentsOfMykin"]
        url_under=source_bs4.select(f'#au_board_list > tr:nth-child({num}) > td.title > a')[0]["href"]
        # url_under=url_under.replace("§","%C2%A7")
       # urls.append(url_under)
    # print(urls)
        with open(f"{detail_address}urls_{name}.txt",'a',encoding='utf-8') as file:
                file.write(f"{page}_{num},")
                file.write(url_under+"\n")
