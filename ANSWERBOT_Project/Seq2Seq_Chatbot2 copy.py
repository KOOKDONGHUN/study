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

# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 67

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든레이어 차원
lstm_hidden_dim = 128

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 챗봇 데이터 로드
chatbot_data = pd.read_csv(r'D:\Study\ANSWERBOT_Project\data\ChatbotData2.csv', encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])