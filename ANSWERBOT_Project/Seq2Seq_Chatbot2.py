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
max_sequences = 67

# 임베딩 벡터 차원
embedding_dim = 100 # 단어 하나가 나타내는 임베딩 차원의 크기?

# LSTM 히든레이어 차원 // 내가 아는건 노드의 개수인데 멘토님은 노드? 항상 이러심 셀이라고 표현하는게 맞을까...?
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

# 질문과 대답 문장들을 하나로 합침 //
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
        words.append(word)

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
            sentence_index.extend([vocabulary[STA]]) # append 안쓰고 extend인 이유? 리스트가 벗겨져서 들어가냐 그냥 들어가냐의 차이?
        
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

        sentences_index.append(sentence_index)

    return np.asarray(sentences_index) # 이미 ndarray가 있다면 복사를 하지 않는다?

# 인코더 입력 인덱스 변환
x_encoder = convert_text_to_index(question, word_to_index, encoder_input)
print(f'x_encoder : {x_encoder[2]}')

# 임의 추가
def invers_index_to_text(x):
    temp = []
    for i in x:
        temp.append(index_to_word[i])
    temp = " ".join(temp)
    print(temp)
invers_index_to_text(x_encoder[2])
# 임의 추가

# 디코더 입력 인덱스 변환
x_decoder = convert_text_to_index(answer, word_to_index, decoder_input)
print(f'x_decoder : {x_decoder[2]}')
invers_index_to_text(x_decoder[2])

# 디코더 목표 인덱스 변환
y_decoder = convert_text_to_index(answer, word_to_index, decoder_target)
print(f'y_decoder : {y_decoder[2]}')
invers_index_to_text(y_decoder[2])

# 원핫인코딩 초기화
one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words)))

# 디코더 목표를 원핫인코딩으로 변환
# 학습시 입력은 인덱스 이지만, 풀력은 원핫 인코딩 형식임
for i, sequence in enumerate(y_decoder):
    for j, index in enumerate(sequence):
        one_hot_data[i, j, index] = 1
    
# 디코더 목표 설정
y_decoder = one_hot_data
y_decoder[2]

# 훈현 모델의 인코더 정의 -----------------------------------

# 입력 문장으로 인덱스로 변환된 시퀀스를 받는다.
print(f'x_encoder.shape : {x_encoder.shape}') # (30, 67)
encoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어 // 원핫을 하지 않고 임베딩을 쓰는 이유는 원핫에는 단어간의 유사도를 따질 수 없음 // 코사인 유사도가 수직
encoder_outputs = layers.Embedding(len(words), embedding_dim)(encoder_inputs)

encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim, # lstm_hidden_dim -> 128 위에 정의됨
                                                    dropout=0.1, 
                                                    recurrent_dropout=0.5,
                                                    return_state=True)(encoder_outputs)

# 히든 상태와 셀 상태를 하나로 묶음
encoder_states = [state_h, state_c]

# 훈련 모델 디코더 정의 -----------------------------------

# 목표 문장의 인덱스 시퀀스를 입력으로 받음
print(f'x_decoder.shape : {x_decoder.shape}') # (30, 67)
decoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
decoder_embedding = layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs)

# 인코더와 다르게 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴 // 타임 스텝을 리턴해준다 ... ?
# 이유가 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
decoder_lstm = layers.LSTM(lstm_hidden_dim, # lstm_hidden_dim -> 128 위에 정의됨
                                dropout=0.1,
                                recurrent_dropout=0.5,
                                return_state=True,
                                return_sequences=True)

# initial_state를 인코더의 상태로 초기화
decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                        initial_state=encoder_states)
print('ㅏㅏㅏㅏ',decoder_outputs)

# 단어의 개수 만큼 노드의 개수를 설정하여 원-핫 형식으로 각 단어 인덱스를 출력 // 여기서 말하는게 형식만 원-핫 인거고 실제로는 임베딩-매트릭스의 인덱스를 찾아 가기 위함?
decoder_dense = layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 훈련 모델 정의 -----------------------------------
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# compile (==학습 방법 설정)
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])
                
model.summary()

# 예측 모델의 인코더 정의 ----------------------------------- // 학습과 예측이 다른 이유 ??
encoder_model = models.Model(encoder_inputs, encoder_states)

# 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행 // 학습은 데이터가 이미 정해져 있지만 예측은 말 그대로 예측 이기 때문에?
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim, )) # 히든 레이어의 디멘션을 맞춰주는 이유가 있는 건가?
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim, ))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 임베딩 레이어
decoder_outputs = decoder_embedding(decoder_inputs)

#LSTM 레이어
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                    initial_state=decoder_states_inputs)

# 히든 상태와 셀 상태를 하나로 묶음
decoder_states = [state_h, state_c]

# Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
decoder_outputs = decoder_dense(decoder_outputs)

# 예측 모델 디코더 설정
decoder_model = models.Model([decoder_inputs] + decoder_states_inputs,
                                [decoder_outputs] + decoder_states)

decoder_model.summary()

# 인덱스를 문장으로 변환 inverse
def convert_index_to_text(indexs, vocabulary):

    sentence = ''

    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == end_index :
            # 종료 인덱스면 중지
            break
        elif vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else :
            # 사전에 없는 인덱스aus OOV 단어를 추가
            sentence += vocabulary[oov_index]

        # 빈칸 추가
        sentence += ' '

    return sentence

for epoch in range(20): # 이거 하는 이유가 예측 때문인가??
    print('Total Epoch :', epoch + 1)
    print(f'x_encoder : {x_encoder.shape}')
    print(f'x_decoder : {x_decoder.shape}')
    print(f'y_decoder : {y_decoder.shape}')
    history = model.fit([x_encoder, x_decoder],
                            y_decoder,
                            epochs=100, # 여기서 epochs를 주는데 위에서 반복문을 쓰는 이유는?
                            batch_size=64,
                            verbose=2)
    # 정확도와 손실 출력
    print('accuracy :', history.history['acc'][-1])
    print('loss :', history.history['loss'][-1])

    # 문장 예측 테스트
    if epoch == 0 :
        print(f'x_encoder[2].shape : {x_encoder[2].shape}')
    input_encoder = x_encoder[2].reshape(1, x_encoder[2].shape[0])
    input_decoder = x_decoder[2].reshape(1, x_decoder[2].shape[0])
    results = model.predict([input_encoder, input_decoder])

    # 결과의 원-핫 인코딩 형식을 인덱스로 변환
    # 1축을 기준으로 가장 높은 값의 위치를 구함
    indexs = np.argmax(results[0], 1)

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
    print(sentence)
    print()

# 모델 저장
encoder_model.save(r'D:\Study\Model\seq2seq_chatbot_encoder_model.h5')
decoder_model.save(r'D:\Study\Model\seq2seq_chatbot_decoder_model.h5')

# 인덱스 저장
with open(r'D:\Study\Model\word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
with open(r'D:\Study\Model\index_to_word.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
# 효율적으로 저장하거나 스트림으로 전송할 때 파이썬 객체의 데이터를 줄로 세워 저장하는 것을 직렬화(serialization) 라고 하고,
# (b) 이렇게 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것을 역직렬화(de-serialization)라고 합니다.

# 모델 파일 로드
encoder_model = models.load_model('./model/seq2seq_chatbot_encoder_model.h5')
decoder_model = models.load_model('./model/seq2seq_chatbot_decoder_model.h5')

# 인덱스 파일 로드
with open(r'D:\Study\Model\word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open(r'D:\Study\Model\index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

def make_predict_input(sentence):

    sentences = []