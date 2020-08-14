# -*- coding: utf-8 -*-
from urllib.request import urlopen
from urllib.parse import quote
import bs4

# name = __file__.split("\\")[-1][:-3]
name = "Certificate2"
num_per_page = range(1,11)#한페이지 10개
pages = range(1,501)# 10개 페이지
detail_address="D:\private\project\main_project\\"



# 2. 긁어온 url 통해서 자료추출

with open(f"{detail_address}urls_{name}.txt",'r') as file:
    lines=file.readlines()
    start=0
    end=3000
    for index,line in enumerate(lines):
    # for index,line in enumerate(lines[start:end]):
        # index+=start
        url=line.split(",")[-1]
        # print(url)
        source = urlopen(url).read()
        source_bs4 = bs4.BeautifulSoup(source,"html.parser")
        question_title_selector_list=['''#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old >div.c-heading__title > div > div.title''',
        '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div',
        '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default > div.c-heading__title > div > div',
         '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--multiple > div.c-heading__title > div > div',
         '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--multiple-old > div.c-heading__content',
         ]
        question_context_selector_list=['''#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content'''
        ]
        # print(source_bs4.select(question_title_selector_list[0])[0].text.strip())
        # print(source_bs4.select('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div'))
        for i in range(0,len(question_title_selector_list)):
            # question_title=source_bs4.select(question_title_selector_list[i])[0].text.strip()
            try:
                
                question_title=source_bs4.select(question_title_selector_list[i])[0].text.strip()
                break
            except:
                if i==len(question_title_selector_list)-1:
                    print(f"error.{line.split(',')[0]}")
                question_title="null"
                
        try:
            question_context=source_bs4.select(question_context_selector_list[0])[0].text.strip()
            # print(context)
        except:
            question_context='null'
        print(index)
        # print("question_title")
        # print(question_title)
        # print("question_context")
        # print(question_context)

        page_dict=dict()

        page_dict["question_title"]=question_title
        page_dict["question_context"]=question_context

        cnt_selector="#answerArea > div.answer-content__inner > div.c-classify.c-classify--sorting > div.c-classify__title-part > h3 > em"
        cnt=source_bs4.select(cnt_selector)[0].text
        cnt=int(cnt)

        answer_writer_selector_list= [f'#answer_{cnt} > div.c-heading-answer > div.c-heading-answer__body > div > p > a',
        f'#answer_{cnt-1} > div.c-heading-answer > div.c-heading-answer__body > div > p > a',
        f'#answer_{cnt} > div.c-heading-answer.c-heading-answer--backout > div.c-heading-answer__body > div.c-heading-answer__title > p',
        f'#answer_1 > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p']
                                    #answer_1 > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p > a
        answer_context_selector_list = [f'#answer_{cnt} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user',
        f'#answer_{cnt-1} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user',
        f'#answer_{cnt-1} > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user> div > div']
                                    #answer_2 > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user            



        try :
            answer_writer = source_bs4.select(answer_writer_selector_list[0])[0].text
            # print("answer_writer")
            # print(answer_writer)
        except:
            # print("test2")
            if cnt!=1:
                try:
                    answer_writer = source_bs4.select(answer_writer_selector_list[1])[0].text
                    # print("test4")
                except:
                    # print("test3")
        
                    answer_writer='null'
            elif cnt==1:
                try:
                    answer_writer = source_bs4.select(answer_writer_selector_list[2])[0].text
                except:
                    try:
                        
                        answer_writer = source_bs4.select(answer_writer_selector_list[3])[0].text
                    except:
                        answer_writer='null'

                
        try :
            answer_context = source_bs4.select(answer_context_selector_list[0])[0].text
        except:
            if cnt!=1:
                try:
                    answer_context = source_bs4.select(answer_context_selector_list[1])[0].text
                except:
                    try:  
                        answer_context = source_bs4.select(answer_context_selector_list[2])[0].text
                    except:
                        answer_context ='null'
        if answer_writer=="null":
            page_dict["answer_writer"]=answer_writer
        else:
            page_dict["answer_writer"]=answer_writer[:-len(" 님 답변")]
        page_dict["answer_context"]=answer_context
        page_dict["url"]=url[:-1]
        page_dict["index"]=line.split(",")[0]
        print("-"*20)
        # for i in page_dict.values():
            # print(i)
        print("-"*20)
        print()

        filenumber=index//100+1
        with open(f"data/crawling_data/a_{filenumber}.txt",'a',encoding='utf-8') as file:
            file.write(str(page_dict))
            file.write(",\n")




