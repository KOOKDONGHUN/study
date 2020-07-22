answer_writer_selector = '#answer_1 > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p > a'

answer_writer_selector = answer_writer_selector.replace(answer_writer_selector[8],'2')
print(answer_writer_selector[8])
print(answer_writer_selector)

ls = list(str(range(1,10)))
ls.append('자격증')

i = 0
while ls[i] != '자격증':
    print('찾기위해 노력중', ls[i])
    i += 1
print(i)