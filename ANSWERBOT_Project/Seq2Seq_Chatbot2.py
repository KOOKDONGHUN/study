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
max_sequences = 30

# 임베딩 벡터 차원
embedding_dim = 100 # 단어 하나가 나타내는 임베딩 차원의 크기?

# LSTM 히든레이어 차원 // 내가 아는건 노드의 개수인데 멘토님은 노드? 항상 이러심 셀이라고 표현하는게 맞을까...
lstm_hidden_dim = 128

# 정규 표현식 필터 // 문장의 특수문자 들을 제거해주기 위함 // 근데 특수문자를 포함해서 학습하면 결과가 안좋아서 그런것인가?
RE_FILTER =  re.compile("[.,!?\":;~()]")

# 데이터 로드
data = pd.read_csv(r'D:\Study\ANSWERBOT_Project\data\ChatbotData2.csv')