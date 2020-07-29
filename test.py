from selenium import webdriver
import time

driver  = webdriver.Chrome("c:/PythonHome/chromedriver.exe")
driver.implicitly_wait(3)

num = 1
page = 33


url = f'https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page={page}'
xpath = f'//*[@id="au_board_list"]/tr[{num}]/td[1]/a'  # xml path language // local path or absolute path?

driver.get(url)
time.sleep(1)

search_res = driver.find_element_by_xpath(xpath).get_attribute('href')
print(search_res)

# driver.quit()