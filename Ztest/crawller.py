# from bs4 import BeautifulSoup
# import urllib.request as req

url = "http://people.incruit.com/resumeguide/pdslist.asp?pds1=1&pds2=11"
url2= 'http://people.incruit.com/resumeguide/pdsview.asp?pds1=1&pds2=11&pdsno=365502&listseq=&page=1&sot=0'

## parser.py
import requests
from bs4 import BeautifulSoup

## HTTP GET Request
req = requests.get(url2)
## HTML 소스 가져오기
html = req.text
## BeautifulSoup으로 html소스를 python객체로 변환하기
## 첫 인자는 html소스코드, 두 번째 인자는 어떤 parser를 이용할지 명시.
## 이 글에서는 Python 내장 html.parser를 이용했다.
soup = BeautifulSoup(html, 'html.parser')
print(soup)

my_titles = soup.select(
    '#detail_info > span.cont > p:nth-child(5)'
    )

print('\n',my_titles)