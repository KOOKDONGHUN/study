from selenium import webdriver
import time

driver  = webdriver.Chrome("c:/PythonHome/chromedriver.exe")
driver.implicitly_wait(3)

num_per_page = list(str(range(1,21)))
page = list(str(range(1,1426)))

data_ls = list()

driver.get('https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page=1')

search_res = driver.find_element_by_xpath('//*[@id="au_board_list"]/tr[2]/td[1]/a')
search_res.click()

driver.switch_to_window(driver.window_handles[1])

qustion = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div')
qustion_detail = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content')

check_num = 1

answer_writer = driver.find_element_by_css_selector('div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p > a')

print(answer_writer.text)
answer_detail = driver.find_element_by_css_selector('div._endContents.c-heading-answer__content')
print(answer_detail.text)
# data_ls.append(answer_writer.text)

driver.close

time.sleep(3)

# file = open("resume.txt",'a',encoding='utf-8')
# for resum in data_ls:
#     file.write(resum+"\n")
# file.close()

driver.quit()

