# from bs4 import BeautifulSoup
# import urllib.request as req

## 10개

# url2 = 

## parser.py
import requests
from bs4 import BeautifulSoup

name_ls = []

## HTTP GET Request
for i in range(1,11):
    url1 = "http://people.incruit.com/resumeguide/pdslist.asp?page="+f"{i}"+"&listseq=1&sot=&pds1=1&pds2=11&pds3=&pds4=&schword=&rschword=&lang=&price=&occu_b_group=&occu_m_group=&occu_s_group=&career=&pass=Y&compty=&rank=&summary=&goodsty="
    req = requests.get(url1)
    html = req.text
    soup1 = BeautifulSoup(html, 'html.parser')
    print(soup1)

    for j in range(1,25):
        selector = "#content > div.bbsWrap > table > tbody > tr:nth-child("+f"{j}"+") > td.numcol" # 24
        my_titles = soup1.select(
            selector
            )
        my_titles = str(my_titles)
        my_titles = my_titles[-12:-6]
        name_ls.append(my_titles)
        print('\n',my_titles)

f = open("./name_ls.txt", 'w')
for i in name_ls:
    data = f"{i}\n"
    f.write(data)
f.close()


# f = open("./name_ls.txt", 'r')
# line = f.readline()
# print(line)
# f.close()

# start = 0
# end = 24

# for k in range(1,11):
#     for l in range(start,end):
#         url2 = "http://people.incruit.com/resumeguide/pdsview.asp?pds1=1&pds2=11&pdsno="+f"{name_ls[l]}"+"&listseq=1&page="+f"{k}"+"&sot=0&pass=Y"
#         req2 = requests.get(url2)
#         html2 = req2.text
#         soup2 = BeautifulSoup(html2, 'html.parser')

#         t_num = 5

#         for k in range(5):
#             selector2 = "#detail_info > span.cont > p:nth-child("+f"{t_num}"+")"
#             my_titles2 = soup2.select(
#                     selector2
#                     )
#             my_titles2 = str(my_titles2)
#             my_titles2 = my_titles2[-30:]
#             # name_ls.append(my_titles)
#             print('\n',my_titles2)
#             t_num += 6
#     start += 24
#     end += 24