import requests
from bs4 import BeautifulSoup

keyword = '동원금속'
url = 'https://search.naver.com/search.naver?where=news&query=' + keyword

raw = requests.get(url,
                   headers={'User-Agent':'Chrome/51.0.2704.103'})
html = BeautifulSoup(raw.text, "html.parser")

articles = html.select("ul.type01 > li")

# 리스트를 사용한 반복문으로 모든 기사에 대해서 제목/언론사 출
with open('./test.csv', 'w') as f:
    for ar in articles:
        title = ar.select_one("a._sp_each_title").text
        source = ar.select_one("span._sp_each_source").text
        line = str(title) + ',' + str(source) + '\n'
        f.write(line)
        print(title, sou