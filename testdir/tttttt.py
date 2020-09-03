from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import re

from konlpy.tag import Kkma

# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝+
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스 정의 ( 단어 사전에 들어갈 인덱스 번호 )
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입 정의 ( 인코딩을 할 것 인지 디코딩을 할 것 인지 convert_text_to_index 함수에 사용 할 인자 값 )
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수 ( 모자르다면 패딩을 채워주고 넘친다면 잘라준다. )
max_sequences = 67

# 임베딩 벡터 차원 ( 형태소 분석기를 통과한 하나의 토큰이 가지는 임베딩 차원의 크기? )
embedding_dim = 100

# LSTM 히든레이어 차원 ( 노드가 128개 )
lstm_hidden_dim = 128 # 2의 배수로 하는게 gpu연산에서 효율성이 올라감 

# 정규 표현식 필터 // 특수 문자를 포함하면 학습에 방해 되나요? // 토큰의 개수 제한? // 메모리 낭비?
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 챗봇 데이터 로드
chatbot_data = pd.read_csv(r'D:\Study\ANSWERBOT_Project\data\ChatbotData2.csv', encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])

# 형태소분석 함수
def pos_tag(sentences):
    
    # KoNLPy 형태소분석기 설정 // 다른 형태소 분석기도 사용해 보자 // 좋습니다 ~~~
    tagger = Kkma()
    
    # 문장 품사 변수 초기화
    sentences_pos = []
    
    # 모든 문장 반복
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)
        
        # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
        sentence = " ".join(tagger.morphs(sentence))
        sentences_pos.append(sentence)
        
    return sentences_pos

# 형태소분석 수행
question = pos_tag(question)
answer = pos_tag(answer)

# 질문과 대답 문장들을 하나로 합침
sentences = []
sentences.extend(question)
sentences.extend(answer)

words = []

# 단어들의 배열 생성 // 위에 형태소 분석 할 때 같이 안 하고 따로 하는 이유? 질문과 대답이 하나로 합쳐져있지 않았기 때문에 ?
for sentence in sentences:
    for word in sentence.split():
        
        if len(word) <= 0:
            print('길이가 0이 있음 !') # // 없는 듯?

        words.append(word)

        # 형태소를 기준으로 한 문장 시퀀스의 최대 길이를 지정 하기 위해 넣어봄
        if max_sequences <= len(sentence.split()):
            max_sequences = len(sentence.split())
            
print(f'max_sequences : {max_sequences}') # max_sequences : 78 // 이렇게 하면 최대 길이인 애들은 마지막 단어가 짤리고 end 태그가 들어감 
max_sequences += 1

#--------------------------------------------
# 훈련 모델 인코더 정의
#--------------------------------------------

# 입력 문장의 인덱스 시퀀스를 입력으로 받음
encoder_inputs = layers.Input(shape=(None,)) # (1425, 79)

# 임베딩 레이어
encoder_outputs = layers.Embedding(len(words), embedding_dim, input_length=max_sequences)(encoder_inputs) # (1425, 79) -> (6406, 100, 78) // 마지막 max_sequences는 자동으로 잡아준다? 생략 가능하다. // 80 넣었을 때 에러가 날까 안날까 확인해보기 
print(f'encoder_outputs : {encoder_outputs}')

# return_state가 True면 상태값 리턴
# LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim, # 128 노드의 개수 
                                                dropout=0.1,
                                                recurrent_dropout=0.5,
                                                return_state=True)(encoder_outputs) # (128, 100, 78)                                                                                         (None, 78, 128)

# 히든 상태와 셀 상태를 하나로 묶음
encoder_states = [state_h, state_c]



#--------------------------------------------
# 훈련 모델 디코더 정의
#--------------------------------------------

# 목표 문장의 인덱스 시퀀스를 입력으로 받음
decoder_inputs = layers.Input(shape=(None,)) # x_decoder : (1425, 78)

# 임베딩 레이어
decoder_embedding = layers.Embedding(len(words), embedding_dim) # (6406, 100, 78)
decoder_outputs = decoder_embedding(decoder_inputs)

# 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
# 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
decoder_lstm = layers.LSTM(lstm_hidden_dim,
                           dropout=0.1,
                           recurrent_dropout=0.5,
                           return_state=True,
                           return_sequences=True) # many to many

# initial_state를 인코더의 상태로 초기화
decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                     initial_state=encoder_states)

# 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
decoder_dense = layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



#--------------------------------------------
# 훈련 모델 정의
#--------------------------------------------

# 입력과 출력으로 함수형 API 모델 생성
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 학습 방법 설정
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])    

model.summary()

#--------------------------------------------
#  예측 모델 인코더 정의
#--------------------------------------------

# 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
encoder_model = models.Model(encoder_inputs, encoder_states)
# 훈련 모델의 인코더의 입력, 훈련 모델의 인코더 lstm에서 나온 스테이트를 출력으로 가지는 예측 인코더 모델 생성


#--------------------------------------------
# 예측 모델 디코더 정의
#--------------------------------------------

# 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
# 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,)) # 이전 타임스템의 스테이트를 받는 부분
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 임베딩 레이어
decoder_outputs = decoder_embedding(decoder_inputs)

# LSTM 레이어
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                 initial_state=decoder_states_inputs)

# 히든 상태와 셀 상태를 하나로 묶음
decoder_states = [state_h, state_c]

# Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
decoder_outputs = decoder_dense(decoder_outputs)

# 예측 모델 디코더 설정
decoder_model = models.Model([decoder_inputs]+decoder_states_inputs, # 입력 2개를 넣는다고 보면 됨
                      [decoder_outputs] + decoder_states)
encoder_model.summary()
decoder_model.summary()