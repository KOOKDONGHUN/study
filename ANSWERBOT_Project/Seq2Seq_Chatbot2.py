from tensorflow import keras
from keras import models, layers, optimizers, losses, metrics
from keras import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os, re

from konlpy.tag import Okt

# Seq2Seq의 동작을 제어하는 태그들 PADDING, START, END, OOV
# START는 디코딩의 시작
# END는 디코딩의 종료
# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스 정의 (단어 사전에 들어갈 인덱스 번호)
pad_indecx = 0
start_index = 1
end_index = 2
oov_index = 3

# 데이터 타입 정의 ?? 인코딩을 할 것인지 디코딩을 할 것인지 디코딩에서도 입력을 넣어줄지 출력을 뽑아낼지 결정
encoder_input = 0
decoder_input = 1
decoder_target = 2

# 한 문장에서 단어의 최대 개수 ex) 안녕하세요 국동훈입니다. == 2 
# 3개라면 앞이나 뒤에 패딩을 추가하여 길이를 맞춰줌 padding 안녕하세요 국동훈입니다.
max_sequences = 0

# 임베딩 벡터 차원
embedding_dim = 100 # 단어 하나가 나타내는 임베딩 차원의 크기?

# LSTM 히든레이어 차원 // 내가 아는건 노드의 개수인데 멘토님은 노드? 항상 이러심 셀이라고 표현하는게 맞을까...
lstm_hidden_dim = 128

# 정규 표현식 필터 // 문장의 특수문자 들을 제거해주기 위함 // 근데 특수문자를 포함해서 학습하면 결과가 안좋아서 그런것인가?
RE_FILTER =  re.compile("[.,!?\":;~()]")

# 데이터 로드
data = pd.read_csv(r'D:\Study\ANSWERBOT_Project\data\ChatbotData2.csv')
print(data)

# 질문과 답변 분리
question, answer = list(data['Q']), list(data['A'])
print(len(question)) # 1425

# 데이터의 일부만 학습에 사용
question = question[:30]
answer = answer[:30]

for i in range(3):
    print(f'Q : {question[i]}')
    print(f'A : {answer[i]}')
    print()

# 단어사전 만들기
def pos_tag(sentences):

    # 형태소 분석기 객체 생성
    tagger = Okt()

    #최종 적으로 return될 형태소로 나뉘어진 문장 데이터
    sentences_pos = []

    # 모든 data(문장)에 대해 반복
    for sentence in sentences:

        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)

        # 배열인 형태소 분석의 출력을 띄어쓰기로 구분하여 붙임
        # tagger.morphs(sentence) // sentence에 대해 형태소 분석을 실시 -> 리스트로 반환
        sentence = tagger.morphs(sentence)

        # " ".join() 리스트 사이사이에 공백을 추가하여 하나의 문장으로 합쳐줌
        sentence = " ".join(sentence)

        # 최종적으로 반환될 데이터 리스트에 추가
        sentences_pos.append(sentence)

    return sentences_pos

# 형태소분석 수행
question = pos_tag(question)
answer = pos_tag(answer)

# 형태소분석으로 변환된 데이터 출력
for i in range(3):
    print('Q : ' + question[i])
    print('A : ' + answer[i])
    print()

# 질문과 대답 문장들을 하나로 합침 // 와닿지는 않음 왜 하는지 잘 모르겠음 // 디코더에 넣을떄 질문 부분을 패딩으로 채워주기 위함?
sentences = []
sentences.extend(question)
sentences.extend(answer)

print('-'*44,'질문과 대답을 하나로 합친 데이터?')
print(f'질문과 대답을 합친 리스트의 차원? {np.shape(sentences)}')
print(sentences[:3])

words = []

# 단어들의 배열을 생성?? 위에 형태소 분석 할 때 같이 안 하고 따로 하는 이유?
for sentence in sentences:
    for word in sentence.split():

        if len(word) <= 0:
            print('길이가 0인 단어? 가 있음 !!!!') # // 없는 듯?

        words.append(word)

        # 형태소를 기준으로 한 문장 시퀀스의 최대 길이를 같이 지정 하기 위해 넣어봄
        if max_sequences <= len(sentence.split()):
            max_sequences = len(sentence.split())

# 길이가 0인 단어는 삭제 // 길이가 0일 수 있나? 
words = [word for word in words if len(word) > 0]

# 중복된 단어 삭제
words = list(set(words))

# 단어 사전 제일 앞에 태그 단어 삽입
words[:0] = [PAD, STA, END, OOV]

print(f'max_sequences : {max_sequences}')

# 단어의 개수
print(len(words)) # 694
print(f'words[:10] : {words[:10]}')

# 단어와 인덱스의 딕셔너리 생성
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}
print(f'word_to_index : {word_to_index}')

# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, type):
    sentences_index = []

    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []

        # 디코더 입력일 경우 맨 앞에 START 태그 추가
        if type == decoder_input:
            sentence_index.extend([vocabulary[STA]]) # append 안쓰고 extend인 이유? 리스트가 벚겨져서 들어가냐 그냥 들어가냐의 차이?
        
        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence.split():
            if vocabulary.get(word) is not None: # 사전에 단어가 있는 경우 인덱스로 변환
                sentence_index.extend([vocabulary[word]])
            else : # 단어 사전에 없는 경우 OOV의 인덱스로 변환
                sentence_index.extend([vocabulary[OOV]])

        # 최대 길이 검사
        if type == decoder_target : # 디코더의 목표(==target==y_train)일 경우 맨 뒤에 END 태그 추가
            if len(sentence_index) >= max_sequences: # 최대 길이보다 길 경우 // 이렇게 되면 최대 길이보다 길다면 마지막은 짤리고 END가 들어가는거 아닌가?
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else : # 최대 길이보다 짧다면 마지막에 END태그
                sentence_index += [vocabulary[END]]
        else : # 디코더의 인풋일 경우 
            # 이전 조건문에서 앞에 START태그를 달아 줬기 때문에 문장의 길이가 최대길이를 초과한다면 아레 조건문을 거치게 되고
            # 마지막을 잘라줌 근데 이거는 초과하지 않아도 잘라줘야 하는거 아닌가?
            if len(sentence_index) > max_sequences: # 디코더의 인풋이 단어 시퀀스의 최대 길이를 넘는 경우
                sentence_index = sentence_index[:max_sequences]

        # 문장이 최대 길이보다 작다면 패딩이 붙을 것이고 똑같다면 패딩은 붙지 않음
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]

    return np.