from selenium import webdriver
import requests
from bs4 import BeautifulSoup
driver  = webdriver.Chrome("c:\\PythonHome\\chromedriver.exe")

#  페이지별로 기업 번호 불러오기
num = list(range(1,21))
for i in num:
    try:
        corp_list = []
        driver.get("http://people.incruit.com/resumeguide/pdslist.asp?page=" +f'{i}'+ "&listseq=1&sot=&pds1=1&pds2=11&pds3=&pds4=&schword=&rschword=&lang=&price=&occu_b_group=&occu_m_group=&occu_s_group=&career=&pass=&compty=&rank=&summary=&goodsty=")
        driver.implicitly_wait(5)
        table = driver.find_element_by_class_name('board_Tbl01')
        tbody = table.find_element_by_tag_name("tbody")
        for i in range(0,24):
            rows = tbody.find_elements_by_tag_name("tr")[i]
            body = rows.find_elements_by_tag_name("td.numcol")
            for index, value in enumerate(body):
                corp_list.append(value.text)
    except:
        continue

for i in corp_list:
    try:
        driver.get("https://people.incruit.com/resumeguide/pdsview.asp?pds1=1&pds2=11&pdsno="+ i +"&listseq=&page=1&sot=0&pass=y")
        # 페이지 소스 가져오기
        # time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html,'lxml')
        resume = soup.find_all(class_='cont')[3].text
        resume = resume.split(".")
        for res in resume:#str
            temp.append(res.replace('\n','').replace('\t','').replace('\r',''))
        file = open("D://resume.txt",'a',encoding='utf-8')
        for resum in temp:
            file.write(resum)
        file.close()
    except:
        continue
driver.close()