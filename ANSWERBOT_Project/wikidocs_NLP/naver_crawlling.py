from selenium import webdriver
import time
import pymssql as ms

def dbconn(tablename, insert_data):
    # conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'INSERT INTO {tablename} values(%s ,%s ,%s ,%s)'
    cursor.execute(sql, (insert_data[0], insert_data[1], insert_data[2], insert_data[3]))
    conn.commit()
    conn.close()

def create_table(tablename):
    # conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tablename}' AND xtype='U')\
         CREATE TABLE {tablename} (id int identity, que text null, que_detail text null,\
             ans_writer text null, ans_detail text null)"
    cursor.execute(sql)
    conn.commit()
    conn.close()

tablename = 'Certificate'

create_table(tablename)

driver  = webdriver.Chrome("c:/PythonHome/chromedriver.exe")
driver.implicitly_wait(3)

num_per_page = range(1,21)
pages = range(1,501) # 1425페이지

# num_per_page = [1]
# num_per_page = [16]
# pages = [118]

for page in pages:
    for num in num_per_page:
        data_ls = list()

        url = f'https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page={page}'
        xpath = f'//*[@id="au_board_list"]/tr[{num}]/td[1]/a'  # xml path language // local path or absolute path?

        question_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div.title'
        question_detail_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content'

        driver.get(url)
        time.sleep(1)

        search_res = driver.find_element_by_xpath(xpath)

        search_res.click()
        time.sleep(1)
        
        driver.switch_to_window(driver.window_handles[1])
        time.sleep(1)

        try :
            question = driver.find_element_by_css_selector(question_selector).text
        except :
            try :
                question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default > div.c-heading__title > div > div').text
            except :
                question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--multiple > div.c-heading__title > div > div').text
                # question = 'Null'
                pass
        # question = question+'\n'

        try :
            question_detail = driver.find_element_by_css_selector(question_detail_selector).text
        except :
            question_detail = 'Null'
            pass
        # question_detail = question_detail+'\n'

        print('check type',type(question))
        data_ls.append(question)

        if len(question_detail) > 900:
            question_detail = question_detail[:900]

        print('check type',type(question_detail))
        data_ls.append(question_detail)

        answer_num = 1

        while True:
            if answer_num >= 15:
                ''' 답변이 삭제되었는지 자격증 따기의 답변이 없음 때문에 답변의 최대 갯수를 15개로 생각하고 15번이상 돌면 더이상 답변을 찾지 않도록함'''
                break

            answer_writer_selector = f'#answer_{answer_num} > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p'
            answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content'

            try :
                answer_writer = driver.find_element_by_css_selector(answer_writer_selector).text
            except:
                pass
            
            try :
                answer_detail = driver.find_element_by_css_selector(answer_detail_selector).text
            except:
                pass

            if answer_writer == '자격증 따기 님 답변' and answer_writer != '비공개 답변':
                answer_detail = answer_detail[:-74]
                if len(answer_detail) > 900 :
                    answer_detail = answer_detail[:900]

                print('check type',type(answer_writer))
                print('check type',type(answer_detail))

                data_ls.append(answer_writer)
                data_ls.append(answer_detail)

                print(f'{num}/20\t{page}-page')
                print(question_detail,'\n')
                print(question,'\n')
                print()
                print(answer_writer,'\n')
                print(answer_detail,'\n')
                print()

                break

            answer_num += 1


        driver.close()
        dbconn(tablename, data_ls)
        time.sleep(1)

        driver.switch_to_window(driver.window_handles[0])
        time.sleep(1)
    
driver.quit()
