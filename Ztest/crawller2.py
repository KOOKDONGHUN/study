import requests
from bs4 import BeautifulSoup

name_ls = []

f = open("./name_ls.txt", 'r')
while True:
    line = f.readline()
    if not line: break
    name_ls.append(line[:-1])
f.close()

print(name_ls)
start = 0
end = 24

for k in range(1,11):
    for l in range(start,end):
        url2 = "http://people.incruit.com/resumeguide/pdsview.asp?pds1=1&pds2=11&pdsno="+f"{name_ls[l]}"+"&listseq=1&page="+f"{k}"+"&sot=0&pass=Y"
        req2 = requests.get(url2)
        html2 = req2.text
        soup2 = BeautifulSoup(html2, 'html.parser')

        t_num = 5

        for k in range(5):
            selector2 = "#detail_info > span.cont > p:nth-child("+f"{t_num}"+")"
            my_titles2 = soup2.select(
                    selector2
                    )
            my_titles2 = str(my_titles2)
            my_titles2 = my_titles2[-100:]
            # name_ls.append(my_titles)
            print('\n',my_titles2)
            t_num += 6
        print('\n\n')
    start += 24
    end += 24