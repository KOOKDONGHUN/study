from selenium import webdriver
import time

driver  = webdriver.Chrome("c:/PythonHome/chromedriver.exe")
driver.implicitly_wait(3)

# 1425페이지
driver.get('https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page=1')
# driver.get('https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page=2')

search_res = driver.find_element_by_xpath('//*[@id="au_board_list"]/tr[1]/td[1]/a')
search_res.click()

driver.switch_to_window(driver.window_handles[1])

# /html/body/div[2]/div[3]/div/div[1]/div[1]/div/div[1]/div[2]/div/div
# //*[@id="content"]/div[1]/div/div[1]/div[2]/div/div
# data = driver.find_element_by_tag_name('html')
# data = data.find_element_by_tag_name('body')
# data = data.find_elements_by_tag_name('div')
# data = data[2].find_element_by_tag_name('div')
# data = data.find_elements_by_tag_name('div')
# data = data.find_elements_by_tag_name('div')[1]
# data = data.find_element_by_tag_name('div')[1]
# data = data.find_elements_by_tag_name('div')
# data = data.find_elements_by_tag_name('div')[1]
# data = data.find_elements_by_tag_name('div')[2]
# data = data.find_element_by_tag_name('div')
# data = data.find_element_by_tag_name('div').text

qustion = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div')
qustion_detail = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content')
answer_writer = driver.find_element_by_css_selector('#answer_1 > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p > a')
answer_detail = driver.find_element_by_css_selector('#answer_1 > div._endContents.c-heading-answer__content')

print(qustion.text)
print(qustion_detail.text)
print(answer_writer.text)
print(answer_detail.text)


# print(data.text)
#페이지당 20개 
# //*[@id="au_board_list"]/tr[1]/td[1]/a
# //*[@id="au_board_list"]/tr[20]/td[1]/a

# 질의 응답페이지
# #content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div  # 질문
# #content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content # 질문 상세 내용
# #answer_1 > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p > a # 답변자 이름
# #answer_1 > div._endContents.c-heading-answer__content # 답변내용

# #content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div
# #content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content
# #answer_3 > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p > a
time.sleep(5)

driver.quit()