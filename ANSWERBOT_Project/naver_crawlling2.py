from selenium import webdriver
import time
import pymssql as ms

def insert_db(tablename, insert_data : dict()):
    # conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'INSERT INTO {tablename} values(%s, %s ,%s ,%s ,%s)'
    cursor.execute(sql, (insert_data['id'], insert_data['que_title'], insert_data['que_detail'], insert_data['ans_writer'], insert_data['ans_detail']))
    conn.commit()
    conn.close()

def create_table(tablename):
    # conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tablename}' AND xtype='U')\
         CREATE TABLE {tablename} (id text null, que text null, que_detail text null,\
             ans_writer text null, ans_detail text null)"
    cursor.execute(sql)
    conn.commit()
    conn.close()

def select_data(tablename):
    # conn = ms.connect(server='192.168.0.176', user='bit2', password='1234',database='bitdb')
    conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'SELECT * FROM {tablename}'
    cursor.execute(sql)
    row = cursor.fetchone()
    while row :
        print(row)
        row = cursor.fetchone()
    conn.close()

# 테이블 명
tablename = 'test1'

# 내가 원하는 답변자
i_want_writer = '자격증 따기 님 답변'

# 크롬드라이버 설치 위치
chrome_path = "c:/PythonHome/chromedriver.exe"

# 경우에 다라 선택자가 달라짐
que_title_selestor_ls = ['div.c-heading._questionContentsArea.c-heading--default-old', # 기본
                         'div.c-heading._questionContentsArea.c-heading--default', # 질문 세부내용이 없는경우
                         'div.c-heading._questionContentsArea.c-heading--multiple'] # 질문 제목은 있지만 세부내용에 사진만 한 경우

# db에 넣을 컬럼명
data_col_name = ['id','que_title', 'que_detail', 'ans_writer','ans_detail']

# 전체 페이지의 개수와 페이지당 개수 입력
num_per_page = range(1,21)
pages = range(351,401)

# num_per_page = range(1,21)
# num_per_page = range(4,21)
# pages = [277]

# db없으면 생성
create_table(tablename)

# 브라우저 생성
driver  = webdriver.Chrome(chrome_path)

# 브라우저가 완전히 열릴때까지 기다리기
driver.implicitly_wait(3)

# 전체 페이지 반복
for page in pages:

    # 페이지당 반복
    for num in num_per_page:

        insert_data = {'id' : f'{page}_{num}',
                       'que_title' : 'Null',
                       'que_detail' : 'Null',
                       'ans_writer' : 'Null',
                        'ans_detail' : 'Null',
                       }

        # 페이지 마다 url이 다름
        url = f'https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page={page}'

        # 페이지에 있는 각 질문들에 대한 번호
        xpath = f'//*[@id="au_board_list"]/tr[{num}]/td[1]/a'  # xml path language // local path or absolute path?

        # 브라우저에 url입력
        driver.get(url)
        time.sleep(1) # 로딩 기다려줌

        # 페이지에 있는 질문에 대한 링크를 클릭
        search_res = driver.find_element_by_xpath(xpath)
        search_res.click()
        time.sleep(1) # 클릭하고 새로운 탭이 열릴때까지 대기
        
        # 새로운 탭으로 이동
        driver.switch_to_window(driver.window_handles[1])
        time.sleep(1) # 이동시간 대기

        # 위에 언급한대로 경우에 따라 질문의 큰 제목이 다름 // 반복
        for selector in que_title_selestor_ls :
            question_selector = f'#content > div.question-content > div > {selector} > div.c-heading__title > div > div.title'
            try :
                insert_data['que_title'] = driver.find_element_by_css_selector(question_selector).text
            except :
                print(f'{num}/20 -- /{page} {selector[-8:]} question Title Error !!!!')
                pass
        
        # 질문 세부내용에 대한 text 추출 이건 에러 발생했다는건 사진만 있거나 내용이 없거나 때문에 그냥 Null 처리
        question_detail_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content'
        try :
            insert_data['que_detail'] = driver.find_element_by_css_selector(question_detail_selector).text
        except :
            print(f'{num}/20 -- /{page} question Detail Error !!!!')
            pass

        # data_ls.append(question)
        # data_ls.append(question_detail)

        # 왜 그런지는 모르겠는데 내가 선택한 답변자의 답변이 없는 경우 무한 반복을 방지하기 위해 answer_num 변수 선언
        answer_num = 1
        while answer_num <= 10:
            
            # 답변자의 이름 선택자
            answer_writer_selector = f'#answer_{answer_num} > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p'

            try :
                insert_data['ans_writer'] = driver.find_element_by_css_selector(answer_writer_selector).text
            except:
                print(f'{num}/20 -- /{page} answer Writer Error !!!!')
                pass
                
            # 답변 내용
            # answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content' # 기존에 사용하던거 뒤에 '알아두세요 이게 출력됨'
            # answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user > div' # 채택된 답변일 경우
            answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user' # 위에꺼는 공백을 가져오네 ㅋ...

            try :
                insert_data['ans_detail'] = driver.find_element_by_css_selector(answer_detail_selector).text
            except:
                print(f'{num}/20 -- /{page} answer Detail Error !!!!')
                pass
            
            # 왜인지는 모르겠으나 비공개 답변인 경우로 답변내용이 추출됨을 방지
            if insert_data['ans_writer'] == i_want_writer and insert_data['ans_writer'] != '비공개 답변':

                # GPT chatbot 예제가 1000자 이하 라서 줄여보려 했는데 일단 수집하고 기계학습 할때 조절하는걸로 
                # if len(answer_detail) > 900 :
                #     answer_detail = answer_detail[:900]

                # data_ls.append(answer_writer)
                # data_ls.append(answer_detail)
                
                print(f'{num}/20\t{page}-page -- {answer_num}번째 답변임')
                for c in data_col_name:
                    print(insert_data[c])
                    print()


                # text 추출이 완료된 시점에서 db에 insert
                insert_db(tablename, insert_data)
                time.sleep(1) # db 삽입 시간 기다려주는게 맞는지 잘 모름 에러는 무서우니 대기

                break # while 탈출 -- 답변에 대한 크롤링 stop

            # while cnt
            answer_num += 1

        # 새로 열린 탭 닫기
        driver.close()
        time.sleep(1)

        # url을 재입력하기 위해 가장 첫번째 탭으로 이동
        driver.switch_to_window(driver.window_handles[0])
        time.sleep(1)
    
driver.quit() # 크롬브라우저 완전 종료