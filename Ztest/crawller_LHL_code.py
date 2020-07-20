from selenium import webdriver
import requests
from bs4 import BeautifulSoup
driver  = webdriver.Chrome("c:/PythonHome/chromedriver.exe")

# #  페이지별로 기업 번호 불러오기
# num = list(range(1,11))
# corp_list = []

# for i in num:
#     try:
        
#         driver.get("http://people.incruit.com/resumeguide/pdslist.asp?page=" +f'{i}'+ "&listseq=1&sot=&pds1=1&pds2=11&pds3=&pds4=&schword=&rschword=&lang=&price=&occu_b_group=&occu_m_group=&occu_s_group=&career=&pass=&compty=&rank=&summary=&goodsty=")
#         driver.implicitly_wait(5)
#         table = driver.find_element_by_class_name('board_Tbl01')
#         tbody = table.find_element_by_tag_name("tbody")
#         for j in range(0,24):
#             rows = tbody.find_elements_by_tag_name("tr")[j]
#             body = rows.find_elements_by_tag_name("td.numcol")
#             for index, value in enumerate(body):
#                 corp_list.append(value.text)
#     except:
#         continue

# print(corp_list)
# print(len(corp_list))
# # driver.quit()
# # driver.close()

start = 0
end = 24

name_ls = []

f = open("./name_ls.txt", 'r')
while True:
    line = f.readline()
    if not line: break
    name_ls.append(line[:-1])
f.close()

for k in range(1,11):
    for i in range(start,end):
        driver.get("https://people.incruit.com/resumeguide/pdsview.asp?pds1=1&pds2=11&pdsno="+ f"{name_ls[i]}" +"&listseq=&page="+f"{k}"+"&sot=0&pass=y")
        https://people.incruit.com/resumeguide/pdsview.asp?pds1=1&pds2=11&pdsno=365502&listseq=&page=1&sot=0&pass=y
        html = driver.page_source
        soup = BeautifulSoup(html,'lxml')
        resume = list(soup.find_all(class_='cont').find_all('p',string=True))
        print(type(resume))
        print(len(resume))
        sentences=[]
        for r in resume:
            k = r.string.split(".")
            for p in k:
                sentences.append(p)
        print(sentences)
        temp=[]
        for res in sentences:#str
            temp.append(res.replace('\n','').replace('\t','').replace('\r',''))
        file = open("resume.txt",'a',encoding='utf-8')
        for resum in temp:
            file.write(resum+"\n")
        file.close()
        # except:
        #     continue
    start += 24
    end += 24
driver.quit()